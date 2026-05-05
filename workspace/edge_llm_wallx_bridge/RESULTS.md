# Edge-LLM + wall-x Bridge Results

## 1. Standard VLM prompt

- Image: `fruits_on_table.png`
- Prompt: `Please describe the image.`

### Qwen2.5-VL-3B-Instruct

- Mean latency: `8194.387 ms`
- Output starts with: `The image depicts a simple, cartoon-style illustration of four fruits...`

### Qwen3-VL-2B-Instruct

- Mean latency: `5645.885 ms`
- Output starts with: `This is a simple, stylized illustration of a face...`

## 2. wall-x style prompt

- Image: `fruits_on_table.png`
- Prompt: `Describe what you see in this image.`
- wall-x reference: `The image features a minimalist composition with a brown background. In the center, there are three distinct shapes`

### Qwen2.5-VL-3B-Instruct

- Wall-clock: `7703.721 ms`
- Output starts with: `The image depicts four colorful objects placed on a flat surface...`
- Text similarity vs wall-x reference: `0.3083`

### Qwen3-VL-2B-Instruct

- Wall-clock: `5457.292 ms`
- Output starts with: `This image is a simple, stylized illustration of a face...`
- Text similarity vs wall-x reference: `0.3169`

## 3. Conclusion

- `TensorRT-Edge-LLM` can run VLM/VQA on Orin.
- Smaller models are faster.
- But on the wall-x style same-image same-prompt case, the default official VLM route does not naturally reproduce the wall-x baseline semantics.
