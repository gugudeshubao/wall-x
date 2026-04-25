"""Quick peak memory check - match Orin test conditions (64 tokens, single image)."""
import os, sys, time, glob, torch
torch.cuda.reset_peak_memory_stats()

from transformers import AutoProcessor
from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import Qwen2_5_VLMoEForAction
from safetensors.torch import load_file
from PIL import Image

model_path = sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/project/github/wall-x/models/wall-oss-flow"
image_path = sys.argv[2] if len(sys.argv) > 2 else None

# Find an image
if image_path is None:
    for d in ["/home/ubuntu/project/github/wall-x/test_images", "/home/ubuntu/project/github/wall-x/images", "/data/wy/wall-x/test_images"]:
        imgs = glob.glob(os.path.join(d, "*.png")) + glob.glob(os.path.join(d, "*.jpg"))
        if imgs:
            image_path = imgs[0]
            break
if image_path is None:
    print("ERROR: no test image found")
    sys.exit(1)

print(f"Model: {model_path}")
print(f"Image: {image_path}")
print(f"Device: {torch.cuda.get_device_name()}")

# Load model
torch.cuda.reset_peak_memory_stats()
config = Qwen2_5_VLMoEForAction.config_class.from_pretrained(model_path)
config._attn_implementation = "flash_attention_2"
processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
model = Qwen2_5_VLMoEForAction(config, processor=processor)
model.resize_token_embeddings(len(processor.tokenizer))
sf_files = glob.glob(os.path.join(model_path, "*.safetensors"))
sd = {}
for f in sf_files:
    sd.update(load_file(f, device="cpu"))
model.load_state_dict(sd, strict=False)
model.eval().to("cuda", dtype=torch.bfloat16)
print(f"After model load: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

# Prepare input - single image, same as Orin test
pil_img = Image.open(image_path).convert("RGB")
print(f"Image size: {pil_img.size}")
messages = [{"role": "user", "content": [{"type": "image", "image": pil_img}, {"type": "text", "text": "What do you see in this image?"}]}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[pil_img], padding=True, return_tensors="pt").to("cuda")
print(f"Input tokens: {inputs['input_ids'].shape[1]}")

# Run with max_new_tokens=64 (same as Orin bench)
torch.cuda.reset_peak_memory_stats()
# Re-measure from this point (model already on GPU, so we measure inference peak)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=64)
torch.cuda.synchronize()
inference_peak = torch.cuda.max_memory_allocated() / 1024**3
out_tokens = out.shape[1] - inputs['input_ids'].shape[1]
print(f"\nmax_new_tokens=64, output tokens: {out_tokens}")
print(f"Peak GPU memory (inference): {inference_peak:.2f} GB")

# Also check total peak (load + inference) without reset
# We need to re-load to get total
print(f"\n--- For reference: total peak including model load ---")
# The peak from load was already printed. The inference peak after reset is above.
# To get comparable number: model weights + KV cache + activations
print(f"Inference peak (model on GPU + generate 64 tokens): {inference_peak:.2f} GB")
