#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch Edge-LLM runtime to log top-k token ids/scores")
    parser.add_argument("--root", required=True, help="TensorRT-Edge-LLM repo root")
    args = parser.parse_args()

    root = Path(args.root)
    target = root / "cpp/runtime/llmInferenceSpecDecodeRuntime.cpp"
    text = target.read_text()

    if '#include <cstdlib>' not in text:
        text = text.replace('#include <functional>\n', '#include <functional>\n#include <cstdlib>\n#include <sstream>\n', 1)

    old_debug = """    if (debugTopK > 1)
    {
        check::check(mSamplingIndices.reshape({activeBatchSize, debugTopK}), \"Tensor reshape failed\");
        check::check(mSamplingScores.reshape({activeBatchSize, debugTopK}), \"Tensor reshape failed\");
        selectAllTopK(mLogitsOutput, std::ref(mSamplingScores), mSamplingIndices, debugTopK, mSamplingWorkspace,
            context.stream);
        check::check(mSamplingIndices.reshape({activeBatchSize, 1}), \"Tensor reshape failed\");
    }
"""
    new_debug = """    if (debugTopK > 1)
    {
        int32_t const effectiveDebugTopK = std::min(debugTopK, mBaseEngineConfig.outputVocabSize);
        int32_t const debugWorkspaceSize = static_cast<int32_t>(
            getSelectAllTopKWorkspaceSize(activeBatchSize, mBaseEngineConfig.outputVocabSize, effectiveDebugTopK));
        rt::Tensor debugWorkspace({debugWorkspaceSize}, rt::DeviceType::kGPU, DataType::kINT8,
            \"LLMInferenceSpecDecodeRuntime::mDebugTopKWorkspace\");
        rt::Tensor debugIndices({activeBatchSize * effectiveDebugTopK}, rt::DeviceType::kGPU, DataType::kINT32,
            \"LLMInferenceSpecDecodeRuntime::mDebugTopKIndices\");
        rt::Tensor debugScores({activeBatchSize * effectiveDebugTopK}, rt::DeviceType::kGPU, DataType::kFLOAT,
            \"LLMInferenceSpecDecodeRuntime::mDebugTopKScores\");
        check::check(debugIndices.reshape({activeBatchSize, effectiveDebugTopK}), \"Tensor reshape failed\");
        check::check(debugScores.reshape({activeBatchSize, effectiveDebugTopK}), \"Tensor reshape failed\");
        selectAllTopK(mLogitsOutput, std::ref(debugScores), debugIndices, effectiveDebugTopK, debugWorkspace,
            context.stream);
        std::vector<float> hostDebugScores(static_cast<size_t>(activeBatchSize * effectiveDebugTopK));
        std::vector<int32_t> hostDebugIndices(static_cast<size_t>(activeBatchSize * effectiveDebugTopK));
        CUDA_CHECK(cudaMemcpyAsync(hostDebugScores.data(), debugScores.rawPointer(),
            static_cast<size_t>(activeBatchSize * effectiveDebugTopK) * sizeof(float), cudaMemcpyDeviceToHost,
            context.stream));
        CUDA_CHECK(cudaMemcpyAsync(hostDebugIndices.data(), debugIndices.rawPointer(),
            static_cast<size_t>(activeBatchSize * effectiveDebugTopK) * sizeof(int32_t), cudaMemcpyDeviceToHost,
            context.stream));
        CUDA_CHECK(cudaStreamSynchronize(context.stream));
        for (int32_t i = 0; i < activeBatchSize; ++i)
        {
            std::ostringstream oss;
            oss << \"[DEBUG_TOPK] batch=\" << i << \" \";
            for (int32_t k = 0; k < effectiveDebugTopK; ++k)
            {
                int32_t const idx = i * effectiveDebugTopK + k;
                if (k > 0)
                {
                    oss << \", \";
                }
                oss << \"(\" << hostDebugIndices[idx] << \",\" << hostDebugScores[idx] << \")\";
            }
            LOG_INFO(\"%s\", oss.str().c_str());
        }
    }
"""
    if old_debug not in text:
        raise SystemExit("debug topk block not found")
    text = text.replace(old_debug, new_debug, 2)

    old_host = """    std::vector<float> hostSelectedScores;
    std::vector<int32_t> hostSelectedTopK;
    if (debugTop1)
    {
        hostSelectedScores.resize(activeBatchSize);
        CUDA_CHECK(cudaMemcpyAsync(hostSelectedScores.data(), mSamplingScores.rawPointer(),
            activeBatchSize * sizeof(float), cudaMemcpyDeviceToHost, context.stream));
    }
    if (debugTopK > 1)
    {
        hostSelectedScores.resize(static_cast<size_t>(activeBatchSize * debugTopK));
        hostSelectedTopK.resize(static_cast<size_t>(activeBatchSize * debugTopK));
        CUDA_CHECK(cudaMemcpyAsync(hostSelectedScores.data(), mSamplingScores.rawPointer(),
            static_cast<size_t>(activeBatchSize * debugTopK) * sizeof(float), cudaMemcpyDeviceToHost, context.stream));
        CUDA_CHECK(cudaMemcpyAsync(hostSelectedTopK.data(), mSamplingIndices.rawPointer(),
            static_cast<size_t>(activeBatchSize * debugTopK) * sizeof(int32_t), cudaMemcpyDeviceToHost, context.stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(context.stream));
    if (debugTop1)
    {
        for (int32_t i = 0; i < activeBatchSize; ++i)
        {
            LOG_INFO(\"[DEBUG_TOP1] token=%d score=%f\", hostSelectedTokenIdsData[i], hostSelectedScores[i]);
        }
    }
    if (debugTopK > 1)
    {
        for (int32_t i = 0; i < activeBatchSize; ++i)
        {
            std::ostringstream oss;
            oss << \"[DEBUG_TOPK] batch=\" << i << \" \";
            for (int32_t k = 0; k < debugTopK; ++k)
            {
                int32_t const idx = i * debugTopK + k;
                if (k > 0)
                {
                    oss << \", \";
                }
                oss << \"(\" << hostSelectedTopK[idx] << \",\" << hostSelectedScores[idx] << \")\";
            }
            LOG_INFO(\"%s\", oss.str().c_str());
        }
    }
"""
    new_host = """    std::vector<float> hostSelectedScores;
    if (debugTop1)
    {
        hostSelectedScores.resize(activeBatchSize);
        CUDA_CHECK(cudaMemcpyAsync(hostSelectedScores.data(), mSamplingScores.rawPointer(),
            activeBatchSize * sizeof(float), cudaMemcpyDeviceToHost, context.stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(context.stream));
    if (debugTop1)
    {
        for (int32_t i = 0; i < activeBatchSize; ++i)
        {
            LOG_INFO(\"[DEBUG_TOP1] token=%d score=%f\", hostSelectedTokenIdsData[i], hostSelectedScores[i]);
        }
    }
"""
    if old_host not in text:
        raise SystemExit("host debug block not found")
    text = text.replace(old_host, new_host, 2)

    target.write_text(text)
    print(f"patched {target}")


if __name__ == "__main__":
    main()
