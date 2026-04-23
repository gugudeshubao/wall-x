"""
Batch VQA inference test for wall-x on RTX 5090.
Runs multiple images with different questions through the model.
"""

import torch
import time
import os
import glob
from PIL import Image
from transformers import AutoProcessor

from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import Qwen2_5_VLMoEForAction


# ──────────────────────────────────────────
# Config
# ──────────────────────────────────────────
MODEL_PATH = "/home/ubuntu/project/models/wall-oss-flow"
TEST_IMAGE_DIR = "/home/ubuntu/project/test_images"
COT_IMAGE = "/home/ubuntu/project/wall-x/assets/cot_example_frame.png"

# (image_path_or_glob, question) pairs
TEST_CASES = [
    # Original cot example
    (
        COT_IMAGE,
        "To move the red block in the plate with same color, what should you do next? Think step by step.",
    ),
    (
        COT_IMAGE,
        "Describe what you see on the table. How many objects are there?",
    ),
    # Generated: blocks and plates
    (
        os.path.join(TEST_IMAGE_DIR, "blocks_and_plates.png"),
        "What objects do you see? Describe their colors and positions.",
    ),
    (
        os.path.join(TEST_IMAGE_DIR, "blocks_and_plates.png"),
        "If I want to place the red cube into the red bowl, what steps should the robot take?",
    ),
    # Generated: robot gripper
    (
        os.path.join(TEST_IMAGE_DIR, "robot_gripper.png"),
        "Describe the robot arm and the objects on the table.",
    ),
    (
        os.path.join(TEST_IMAGE_DIR, "robot_gripper.png"),
        "Which object should the robot pick up first to sort them by color?",
    ),
    # Generated: stacking task
    (
        os.path.join(TEST_IMAGE_DIR, "stacking_task.png"),
        "There is a stack of blocks on the right. Describe the stacking order from bottom to top.",
    ),
    (
        os.path.join(TEST_IMAGE_DIR, "stacking_task.png"),
        "How would you stack the loose blocks on the left to match the tower on the right?",
    ),
    # Generated: dual arm
    (
        os.path.join(TEST_IMAGE_DIR, "dual_arm_robot.png"),
        "Describe the scene. What task could the two robot arms collaborate on?",
    ),
    # Generated: fruits
    (
        os.path.join(TEST_IMAGE_DIR, "fruits_on_table.png"),
        "What fruits are on the table? Move them onto the white plate.",
    ),
    # Real photos (random from picsum, general questions)
    (
        os.path.join(TEST_IMAGE_DIR, "real_tabletop_1.jpg"),
        "Describe what you see in this image in detail.",
    ),
    (
        os.path.join(TEST_IMAGE_DIR, "real_tabletop_2.jpg"),
        "What objects are in this image? Could a robot interact with them?",
    ),
    (
        os.path.join(TEST_IMAGE_DIR, "real_tabletop_3.jpg"),
        "Describe this scene. What actions could be performed here?",
    ),
]


def load_model(model_path: str):
    """Load model and processor once."""
    print(f"Loading model from {model_path} ...")
    t0 = time.time()
    model = Qwen2_5_VLMoEForAction.from_pretrained(model_path)
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda", dtype=torch.bfloat16)
        device = "cuda"
    else:
        device = "cpu"
    dt = time.time() - t0
    print(f"Model loaded in {dt:.1f}s on {device}")
    print(f"GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    return model, device


def run_vqa(model, device, image: Image.Image, question: str, max_new_tokens=512):
    """Run a single VQA inference."""
    processor = model.processor

    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": question}],
        }
    ]
    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text_prompt], images=[image], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    generation_params = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "eos_token_id": processor.tokenizer.eos_token_id,
        "pad_token_id": processor.tokenizer.pad_token_id
        if processor.tokenizer.pad_token_id is not None
        else processor.tokenizer.eos_token_id,
    }

    t0 = time.time()
    with torch.no_grad():
        generated_ids = model.generate(**inputs, **generation_params)
    dt = time.time() - t0

    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return response, dt


def main():
    model, device = load_model(MODEL_PATH)

    print("\n" + "=" * 80)
    print("BATCH VQA INFERENCE TEST")
    print("=" * 80)

    results = []
    total_time = 0.0

    for i, (img_path, question) in enumerate(TEST_CASES):
        print(f"\n{'─' * 80}")
        print(f"[{i+1}/{len(TEST_CASES)}] Image: {os.path.basename(img_path)}")
        print(f"  Q: {question}")

        if not os.path.exists(img_path):
            print(f"  ⚠️  Image not found, skipping.")
            results.append({"status": "SKIP", "image": img_path})
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            answer, dt = run_vqa(model, device, img, question)
            total_time += dt
            print(f"  A: {answer}")
            print(f"  ⏱  {dt:.2f}s | GPU mem: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
            results.append({"status": "OK", "image": img_path, "time": dt})
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            results.append({"status": "FAIL", "image": img_path, "error": str(e)})

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    ok = sum(1 for r in results if r["status"] == "OK")
    fail = sum(1 for r in results if r["status"] == "FAIL")
    skip = sum(1 for r in results if r["status"] == "SKIP")
    times = [r["time"] for r in results if r.get("time")]

    print(f"  Total: {len(results)} | OK: {ok} | FAIL: {fail} | SKIP: {skip}")
    if times:
        print(f"  Total inference time: {total_time:.2f}s")
        print(f"  Avg per image: {total_time/len(times):.2f}s")
        print(f"  Min: {min(times):.2f}s | Max: {max(times):.2f}s")
    print(f"  Peak GPU memory: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")


if __name__ == "__main__":
    main()
