// =============================================================================
// CUTLASS Fused INT8 GEMM + Dequantization Epilogue for SM 8.x (Ampere)
//
// Key optimization: INT32 accumulator stays in registers — scale multiplication
// and BF16 conversion happen in the epilogue, avoiding INT32 writeback to GMEM.
//
// EVT tree:
//   Store[BF16] ← Multiply ← (Multiply ← (AccFetch, ColBroadcast[act_scale]),
//                               RowBroadcast[weight_scale])
// =============================================================================

#include "kernels/cutlass_int8_gemm.h"

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/default_gemm_universal_with_visitor.h>
#include <cutlass/epilogue/threadblock/fusion/visitors.hpp>
#include <cutlass/epilogue/threadblock/epilogue_with_visitor_callbacks.h>

#include <ATen/cuda/CUDAContext.h>
#include <iostream>

using namespace cute;

// =====================================================================
// Type aliases
// =====================================================================
using ElementA          = int8_t;
using LayoutA           = cutlass::layout::RowMajor;
using ElementB          = int8_t;
using LayoutB           = cutlass::layout::ColumnMajor;
using ElementOutput     = cutlass::bfloat16_t;
using LayoutC           = cutlass::layout::RowMajor;
using ElementAccumulator = int32_t;
using ElementCompute    = float;

constexpr int AlignmentA = 16;   // 128 bits / 8 bits
constexpr int AlignmentB = 16;
constexpr int AlignmentC = 8;    // 128 bits / 16 bits (bf16)

using ArchTag       = cutlass::arch::Sm80;
using OperatorClass = cutlass::arch::OpClassTensorOp;
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;
using WarpShape        = cutlass::gemm::GemmShape<64, 64, 64>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
constexpr int NumStages         = 4;
constexpr int EVTEpilogueStages = 1;

// =====================================================================
// EVT (Epilogue Visitor Tree) definition
//
// D[m,n] = Acc[m,n] * act_scale[m] * weight_scale[n] + bias[n]
// Output dtype: BF16
// =====================================================================
using OutputTileThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
    ThreadblockShape, WarpShape, ElementOutput, AlignmentC, EVTEpilogueStages>;

// Leaf: fetch INT32 accumulator
using AccFetch = cutlass::epilogue::threadblock::VisitorAccFetch;

// Leaf: load act_scale[M] per-row (ColBroadcast = per-row vector broadcast to all cols)
using ActScaleLoad = cutlass::epilogue::threadblock::VisitorColBroadcast<
    OutputTileThreadMap, float, Stride<_1, _0, _0>>;

// Leaf: load weight_scale[N] per-col (RowBroadcast = per-col vector broadcast to all rows)
using WtScaleLoad = cutlass::epilogue::threadblock::VisitorRowBroadcast<
    OutputTileThreadMap, float, Stride<_0, _1, int32_t>>;

// Compute: Acc * act_scale → float
using MulActScale = cutlass::epilogue::threadblock::VisitorCompute<
    cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;

// Compute: (Acc * act_scale) * weight_scale → float
using MulWtScale = cutlass::epilogue::threadblock::VisitorCompute<
    cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;

// Optional bias add: + bias[n]
using BiasLoad = cutlass::epilogue::threadblock::VisitorRowBroadcast<
    OutputTileThreadMap, ElementOutput, Stride<_0, _1, int32_t>>;

using AddBias = cutlass::epilogue::threadblock::VisitorCompute<
    cutlass::plus, float, float, cutlass::FloatRoundStyle::round_to_nearest>;

// Store: float → BF16
using StoreD = cutlass::epilogue::threadblock::VisitorAuxStore<
    OutputTileThreadMap, ElementOutput,
    cutlass::FloatRoundStyle::round_to_nearest,
    Stride<int64_t, _1, int64_t>>;

// Compose the tree: Store( Mul( Mul(Acc, ActScale), WtScale ) )
using EVT_AccMulAct = cutlass::epilogue::threadblock::Sm80EVT<
    MulActScale, AccFetch, ActScaleLoad>;

using EVT_MulBoth = cutlass::epilogue::threadblock::Sm80EVT<
    MulWtScale, EVT_AccMulAct, WtScaleLoad>;

using EVT_NoBias = cutlass::epilogue::threadblock::Sm80EVT<StoreD, EVT_MulBoth>;

using EVT_AddBias = cutlass::epilogue::threadblock::Sm80EVT<
    AddBias, EVT_MulBoth, BiasLoad>;

using EVT_WithBias = cutlass::epilogue::threadblock::Sm80EVT<StoreD, EVT_AddBias>;

// =====================================================================
// GEMM kernel type
// =====================================================================
using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
    ElementA, LayoutA, cutlass::ComplexTransform::kNone, AlignmentA,
    ElementB, LayoutB, cutlass::ComplexTransform::kNone, AlignmentB,
    ElementOutput, LayoutC, AlignmentC,
    ElementAccumulator,
    ElementCompute,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EVT_NoBias,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    NumStages,
    cutlass::arch::OpMultiplyAddSaturate,   // INT8 requires Saturate variant
    EVTEpilogueStages
>::GemmKernel;

using GemmDevice = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using GemmKernelBias = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
    ElementA, LayoutA, cutlass::ComplexTransform::kNone, AlignmentA,
    ElementB, LayoutB, cutlass::ComplexTransform::kNone, AlignmentB,
    ElementOutput, LayoutC, AlignmentC,
    ElementAccumulator,
    ElementCompute,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EVT_WithBias,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    NumStages,
    cutlass::arch::OpMultiplyAddSaturate,
    EVTEpilogueStages
>::GemmKernel;

using GemmDeviceBias = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelBias>;

template <typename GemmDeviceT>
static void run_gemm_device(
    typename GemmDeviceT::Arguments& args,
    const torch::Tensor& tensor_for_options) {
    GemmDeviceT gemm_device;

    size_t workspace_size = GemmDeviceT::get_workspace_size(args);
    auto workspace = workspace_size > 0
        ? torch::empty({static_cast<int64_t>(workspace_size)},
                       tensor_for_options.options().dtype(torch::kUInt8))
        : torch::Tensor();

    auto status = gemm_device.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error(
            std::string("CUTLASS cannot implement: ") +
            cutlass::cutlassGetStatusString(status));
    }

    status = gemm_device.initialize(
        args,
        workspace_size > 0 ? workspace.data_ptr() : nullptr,
        at::cuda::getCurrentCUDAStream());

    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error(
            std::string("CUTLASS init failed: ") +
            cutlass::cutlassGetStatusString(status));
    }

    status = gemm_device.run(at::cuda::getCurrentCUDAStream());

    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error(
            std::string("CUTLASS run failed: ") +
            cutlass::cutlassGetStatusString(status));
    }
}


// =====================================================================
// Wrapper function
// =====================================================================
namespace cutlass_int8 {

torch::Tensor gemm_dequant(
    const torch::Tensor& act_int8,       // [M, K] int8
    const torch::Tensor& weight_int8,    // [N, K] int8 (NOT transposed)
    const torch::Tensor& act_scale,      // [M] f32
    const torch::Tensor& weight_scale,   // [N] f32
    const torch::Tensor& bias)           // [N] bf16 optional
{
    TORCH_CHECK(act_int8.is_cuda() && act_int8.dtype() == torch::kInt8);
    TORCH_CHECK(weight_int8.is_cuda() && weight_int8.dtype() == torch::kInt8);
    TORCH_CHECK(!bias.defined() || (bias.is_cuda() && bias.dtype() == torch::kBFloat16),
                "bias must be CUDA bf16 when provided");

    const int M = act_int8.size(0);
    const int K = act_int8.size(1);
    const int N = weight_int8.size(0);

    auto output = torch::empty({M, N},
        act_int8.options().dtype(torch::kBFloat16));

    // --- GEMM arguments ---
    cutlass::gemm::GemmCoord problem_size(M, N, K);
    if (bias.defined()) {
        typename EVT_WithBias::Arguments evt_args{
            {
                {
                    {
                        {},
                        {act_scale.data_ptr<float>(), 1.0f, {}},
                        {}
                    },
                    {weight_scale.data_ptr<float>(), 1.0f, {_0{}, _1{}, int32_t(N)}},
                    {}
                },
                {reinterpret_cast<ElementOutput const*>(bias.data_ptr()),
                 ElementOutput(1.0f), {_0{}, _1{}, int32_t(N)}},
                {}
            },
            {reinterpret_cast<ElementOutput*>(output.data_ptr()),
             {(int64_t)N, _1{}, (int64_t)M * N}}
        };

        typename GemmDeviceBias::Arguments args(
            cutlass::gemm::GemmUniversalMode::kGemm,
            problem_size,
            1,
            evt_args,
            act_int8.data_ptr<int8_t>(),
            weight_int8.data_ptr<int8_t>(),
            nullptr,
            nullptr,
            (int64_t)M * K,
            (int64_t)N * K,
            0,
            0,
            K,
            K,
            N,
            N
        );

        run_gemm_device<GemmDeviceBias>(args, act_int8);
    } else {
        typename EVT_NoBias::Arguments evt_args{
            {
                {
                    {},
                    {act_scale.data_ptr<float>(), 1.0f, {}},
                    {}
                },
                {weight_scale.data_ptr<float>(), 1.0f, {_0{}, _1{}, int32_t(N)}},
                {}
            },
            {reinterpret_cast<ElementOutput*>(output.data_ptr()),
             {(int64_t)N, _1{}, (int64_t)M * N}}
        };

        typename GemmDevice::Arguments args(
            cutlass::gemm::GemmUniversalMode::kGemm,
            problem_size,
            1,
            evt_args,
            act_int8.data_ptr<int8_t>(),
            weight_int8.data_ptr<int8_t>(),
            nullptr,
            nullptr,
            (int64_t)M * K,
            (int64_t)N * K,
            0,
            0,
            K,
            K,
            N,
            N
        );

        run_gemm_device<GemmDevice>(args, act_int8);
    }

    return output;
}

}  // namespace cutlass_int8
