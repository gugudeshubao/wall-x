#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def ensure_include(text: str) -> str:
    needle = "#include <functional>\n"
    extra = "#include <cstdlib>\n"
    if extra in text:
        return text
    if needle not in text:
        raise RuntimeError("could not find include insertion point")
    return text.replace(needle, needle + extra, 1)


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise RuntimeError(f"{label}: target block not found")
    return text.replace(old, new, 1)


def replace_n(text: str, old: str, new: str, n: int, label: str) -> str:
    count = text.count(old)
    if count < n:
        raise RuntimeError(f"{label}: expected at least {n} matches, got {count}")
    return text.replace(old, new, n)


def patch_runtime_cpp(text: str) -> str:
    text = ensure_include(text)

    prefill_block = """    if (shouldUseNonGreedySampling(context.temperature, context.topK, context.topP))
    {
        SamplingParams params(activeBatchSize, mBaseEngineConfig.outputVocabSize, context.temperature,
            static_cast<int32_t>(context.topK), context.topP);
        topKtopPSamplingFromLogits(mLogitsOutput, mSamplingIndices, params, mSamplingWorkspace, context.stream);
    }
    else
    {
        constexpr int32_t kSAMPLING_TOP_K = 1;
        selectAllTopK(
            mLogitsOutput, std::nullopt, mSamplingIndices, kSAMPLING_TOP_K, mSamplingWorkspace, context.stream);
    }

    // Apply vocabulary mapping if base model uses reduced vocabulary
"""
    prefill_repl = """    bool const debugTop1 = std::getenv("EDGELLM_DEBUG_TOP1") != nullptr;
    if (shouldUseNonGreedySampling(context.temperature, context.topK, context.topP))
    {
        SamplingParams params(activeBatchSize, mBaseEngineConfig.outputVocabSize, context.temperature,
            static_cast<int32_t>(context.topK), context.topP);
        topKtopPSamplingFromLogits(mLogitsOutput, mSamplingIndices, params, mSamplingWorkspace, context.stream);
    }
    else
    {
        constexpr int32_t kSAMPLING_TOP_K = 1;
        if (debugTop1)
        {
            check::check(mSamplingScores.reshape({activeBatchSize, 1}), "Tensor reshape failed");
            selectAllTopK(
                mLogitsOutput, std::ref(mSamplingScores), mSamplingIndices, kSAMPLING_TOP_K, mSamplingWorkspace, context.stream);
        }
        else
        {
            selectAllTopK(
                mLogitsOutput, std::nullopt, mSamplingIndices, kSAMPLING_TOP_K, mSamplingWorkspace, context.stream);
        }
    }

    // Apply vocabulary mapping if base model uses reduced vocabulary
"""
    text = replace_once(text, prefill_block, prefill_repl, "prefill sampling block")

    decode_block = """    if (shouldUseNonGreedySampling(context.temperature, context.topK, context.topP))
    {
        SamplingParams params(activeBatchSize, mBaseEngineConfig.outputVocabSize, context.temperature,
            static_cast<int32_t>(context.topK), context.topP);
        topKtopPSamplingFromLogits(mLogitsOutput, mSamplingIndices, params, mSamplingWorkspace, context.stream);
    }
    else
    {
        // Greedy decoding (temperature ~= 0 or default)
        constexpr int32_t kSAMPLING_TOP_K = 1;
        selectAllTopK(
            mLogitsOutput, std::nullopt, mSamplingIndices, kSAMPLING_TOP_K, mSamplingWorkspace, context.stream);
    }

    // Apply vocabulary mapping if base model uses reduced vocabulary
"""
    decode_repl = """    bool const debugTop1 = std::getenv("EDGELLM_DEBUG_TOP1") != nullptr;
    if (shouldUseNonGreedySampling(context.temperature, context.topK, context.topP))
    {
        SamplingParams params(activeBatchSize, mBaseEngineConfig.outputVocabSize, context.temperature,
            static_cast<int32_t>(context.topK), context.topP);
        topKtopPSamplingFromLogits(mLogitsOutput, mSamplingIndices, params, mSamplingWorkspace, context.stream);
    }
    else
    {
        // Greedy decoding (temperature ~= 0 or default)
        constexpr int32_t kSAMPLING_TOP_K = 1;
        if (debugTop1)
        {
            check::check(mSamplingScores.reshape({activeBatchSize, 1}), "Tensor reshape failed");
            selectAllTopK(
                mLogitsOutput, std::ref(mSamplingScores), mSamplingIndices, kSAMPLING_TOP_K, mSamplingWorkspace, context.stream);
        }
        else
        {
            selectAllTopK(
                mLogitsOutput, std::nullopt, mSamplingIndices, kSAMPLING_TOP_K, mSamplingWorkspace, context.stream);
        }
    }

    // Apply vocabulary mapping if base model uses reduced vocabulary
"""
    text = replace_once(text, decode_block, decode_repl, "decode sampling block")

    copy_block = """    CUDA_CHECK(cudaMemcpyAsync(hostSelectedTokenIdsData, mSamplingIndices.rawPointer(),
        activeBatchSize * sizeof(int32_t), cudaMemcpyDeviceToHost, context.stream));
    CUDA_CHECK(cudaStreamSynchronize(context.stream));

    // Update tokenIds and generation length for each sequence
"""
    copy_repl = """    CUDA_CHECK(cudaMemcpyAsync(hostSelectedTokenIdsData, mSamplingIndices.rawPointer(),
        activeBatchSize * sizeof(int32_t), cudaMemcpyDeviceToHost, context.stream));
    std::vector<float> hostSelectedScores;
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
            LOG_INFO("[DEBUG_TOP1] token=%d score=%f", hostSelectedTokenIdsData[i], hostSelectedScores[i]);
        }
    }

    // Update tokenIds and generation length for each sequence
"""
    text = replace_n(text, copy_block, copy_repl, 2, "host copy blocks")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch Edge-LLM runtime to log first-token top1 ids/scores")
    parser.add_argument("--root", required=True, help="TensorRT-Edge-LLM repo root")
    args = parser.parse_args()

    root = Path(args.root)
    target = root / "cpp/runtime/llmInferenceSpecDecodeRuntime.cpp"
    text = target.read_text()
    text = patch_runtime_cpp(text)
    target.write_text(text)
    print(f"patched {target}")


if __name__ == "__main__":
    main()
