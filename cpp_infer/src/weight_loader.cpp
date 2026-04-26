#include "weight_loader.h"
#include <fstream>
#include <iostream>
#include <filesystem>
#include <cstring>
#include <sstream>

namespace fs = std::filesystem;

// Safetensors format:
// [8 bytes: header_size as uint64_t LE]
// [header_size bytes: JSON header]
// [remaining bytes: raw tensor data]
//
// JSON header maps tensor_name -> { "dtype": "BF16", "shape": [x,y,...],
//                                    "data_offsets": [start, end] }

static torch::ScalarType parse_dtype(const std::string& dtype_str) {
    if (dtype_str == "BF16" || dtype_str == "bf16") return torch::kBFloat16;
    if (dtype_str == "F16" || dtype_str == "fp16") return torch::kFloat16;
    if (dtype_str == "F32" || dtype_str == "fp32") return torch::kFloat32;
    if (dtype_str == "I32" || dtype_str == "int32") return torch::kInt32;
    if (dtype_str == "I64" || dtype_str == "int64") return torch::kInt64;
    if (dtype_str == "U8") return torch::kUInt8;
    if (dtype_str == "I8") return torch::kInt8;
    if (dtype_str == "BOOL") return torch::kBool;
    throw std::runtime_error("Unsupported dtype: " + dtype_str);
}

static int dtype_size(torch::ScalarType dtype) {
    switch (dtype) {
        case torch::kBFloat16:
        case torch::kFloat16: return 2;
        case torch::kFloat32: return 4;
        case torch::kInt32: return 4;
        case torch::kInt64: return 8;
        case torch::kUInt8:
        case torch::kInt8:
        case torch::kBool: return 1;
        default: return 4;
    }
}

// Minimal JSON parser for safetensors header (no external dependency)
// This parses the header to extract tensor metadata
struct TensorMeta {
    std::string dtype;
    std::vector<int64_t> shape;
    int64_t data_start;
    int64_t data_end;
};

// Simple JSON string extraction helper
static std::string extract_string(const std::string& json, const std::string& key) {
    auto pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos) return "";
    pos = json.find(":", pos);
    if (pos == std::string::npos) return "";
    pos = json.find("\"", pos);
    if (pos == std::string::npos) return "";
    pos++;
    auto end = json.find("\"", pos);
    return json.substr(pos, end - pos);
}

// Extract array of integers
static std::vector<int64_t> extract_int_array(const std::string& json, const std::string& key) {
    std::vector<int64_t> result;
    auto pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos) return result;
    pos = json.find("[", pos);
    if (pos == std::string::npos) return result;
    auto end = json.find("]", pos);
    std::string arr = json.substr(pos + 1, end - pos - 1);

    // Parse comma-separated integers
    std::stringstream ss(arr);
    std::string item;
    while (std::getline(ss, item, ',')) {
        // Trim whitespace
        item.erase(0, item.find_first_not_of(" \t\n\r"));
        item.erase(item.find_last_not_of(" \t\n\r") + 1);
        if (!item.empty()) {
            result.push_back(std::stoll(item));
        }
    }
    return result;
}

WeightMap load_safetensors(const std::string& path, torch::Device device) {
    WeightMap weights;

    // Read entire file
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open safetensors file: " + path);
    }
    size_t file_size = file.tellg();
    file.seekg(0);

    // Read header size (8 bytes, little-endian uint64)
    uint64_t header_size = 0;
    file.read(reinterpret_cast<char*>(&header_size), 8);

    // Read JSON header
    std::string header(header_size, '\0');
    file.read(&header[0], header_size);

    size_t data_offset = 8 + header_size;

    // Read all tensor data into CPU buffer first
    size_t data_size = file_size - data_offset;
    std::vector<char> data_buffer(data_size);
    file.read(data_buffer.data(), data_size);
    file.close();

    // Parse header: find all tensor entries
    // Header format: { "tensor_name": { "dtype": "BF16", "shape": [...],
    //                   "data_offsets": [start, end] }, ... }
    // We parse it by finding tensor name blocks

    size_t pos = 0;
    while (pos < header.size()) {
        // Find next tensor name (skip __metadata__)
        auto name_start = header.find("\"", pos);
        if (name_start == std::string::npos) break;
        name_start++;
        auto name_end = header.find("\"", name_start);
        if (name_end == std::string::npos) break;
        std::string name = header.substr(name_start, name_end - name_start);

        // Skip __metadata__
        if (name == "__metadata__") {
            // Skip to next top-level entry
            auto brace = header.find("{", name_end);
            if (brace != std::string::npos) {
                int depth = 1;
                size_t i = brace + 1;
                while (i < header.size() && depth > 0) {
                    if (header[i] == '{') depth++;
                    if (header[i] == '}') depth--;
                    i++;
                }
                pos = i;
            } else {
                pos = name_end + 1;
            }
            continue;
        }

        // Find the tensor's object block
        auto obj_start = header.find("{", name_end);
        if (obj_start == std::string::npos) break;
        auto obj_end = header.find("}", obj_start);
        if (obj_end == std::string::npos) break;
        std::string obj = header.substr(obj_start, obj_end - obj_start + 1);

        // Extract metadata
        std::string dtype_str = extract_string(obj, "dtype");
        auto shape = extract_int_array(obj, "shape");
        auto offsets = extract_int_array(obj, "data_offsets");

        if (!dtype_str.empty() && offsets.size() == 2) {
            torch::ScalarType dtype = parse_dtype(dtype_str);
            int64_t start = offsets[0];
            int64_t end = offsets[1];

            // Create tensor from raw data
            auto options = torch::TensorOptions().dtype(dtype);
            torch::Tensor cpu_tensor = torch::from_blob(
                data_buffer.data() + start,
                shape,
                options
            ).clone();  // clone to own the data

            // Move to device
            weights[name] = cpu_tensor.to(device);
        }

        pos = obj_end + 1;
    }

    std::cout << "[WeightLoader] Loaded " << weights.size()
              << " tensors from " << path << std::endl;
    return weights;
}

WeightMap load_weights_from_dir(const std::string& dir_path, torch::Device device) {
    WeightMap weights;

    for (const auto& entry : fs::directory_iterator(dir_path)) {
        if (entry.path().extension() == ".safetensors") {
            auto file_weights = load_safetensors(entry.path().string(), device);
            weights.insert(file_weights.begin(), file_weights.end());
        }
    }

    if (weights.empty()) {
        // Try single model.safetensors
        std::string single = dir_path + "/model.safetensors";
        if (fs::exists(single)) {
            weights = load_safetensors(single, device);
        }
    }

    std::cout << "[WeightLoader] Total: " << weights.size()
              << " tensors from " << dir_path << std::endl;
    return weights;
}

void print_weight_names(const WeightMap& weights) {
    std::vector<std::string> names;
    names.reserve(weights.size());
    for (const auto& [name, tensor] : weights) {
        names.push_back(name);
    }
    std::sort(names.begin(), names.end());
    for (const auto& name : names) {
        const auto& t = weights.at(name);
        std::cout << "  " << name << " : " << t.sizes() << " " << t.dtype() << std::endl;
    }
}
