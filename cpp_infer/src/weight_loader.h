#pragma once

#include <torch/torch.h>
#include <string>
#include <unordered_map>
#include <vector>

using WeightMap = std::unordered_map<std::string, torch::Tensor>;

// Load model weights from safetensors format
// Returns a map of tensor name -> tensor (on specified device)
WeightMap load_safetensors(const std::string& path, torch::Device device);

// Load weights from a directory containing one or more safetensors files
// Merges all files into a single weight map
WeightMap load_weights_from_dir(const std::string& dir_path, torch::Device device);

// Dump weight names for debugging
void print_weight_names(const WeightMap& weights);
