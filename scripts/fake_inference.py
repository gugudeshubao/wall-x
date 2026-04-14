import argparse
import os

import torch

from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import Qwen2_5_VLMoEForAction


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default=os.environ.get("WALL_X_MODEL_PATH", "/path/to/model"),
        help="Path to the downloaded Wall-X model directory.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-length", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.model_path == "/path/to/model" or not os.path.exists(args.model_path):
        raise FileNotFoundError(
            f"Model path does not exist: {args.model_path}. "
            "Pass --model-path or set WALL_X_MODEL_PATH."
        )

    model = Qwen2_5_VLMoEForAction.from_pretrained(args.model_path)
    model.eval()

    batch_size = args.batch_size
    seq_length = args.seq_length

    torch.manual_seed(0)
    fake_input_ids = torch.randint(
        0, len(model.processor.tokenizer), (batch_size, seq_length), dtype=torch.long
    )
    fake_attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
    fake_moe_token_types = torch.zeros((batch_size, seq_length), dtype=torch.long)
    fake_position_ids = (
        torch.arange(seq_length, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    )
    fake_proprioception = torch.randn((batch_size, 1, 20), dtype=torch.float32)
    fake_agent_pos_mask = torch.ones((batch_size, 1, 20), dtype=torch.float32)
    fake_dof_mask = torch.ones((batch_size, 32, 20), dtype=torch.float32)
    fake_dataset_names = ["x2_normal"]

    if args.device == "cuda":
        model = model.to(args.device).bfloat16()
        fake_proprioception = fake_proprioception.to(args.device).bfloat16()
        fake_agent_pos_mask = fake_agent_pos_mask.to(args.device).bfloat16()
        fake_dof_mask = fake_dof_mask.to(args.device).bfloat16()
    else:
        model = model.to(args.device)
        fake_proprioception = fake_proprioception.to(args.device)
        fake_agent_pos_mask = fake_agent_pos_mask.to(args.device)
        fake_dof_mask = fake_dof_mask.to(args.device)

    fake_input_ids = fake_input_ids.to(args.device)
    fake_attention_mask = fake_attention_mask.to(args.device)
    fake_moe_token_types = fake_moe_token_types.to(args.device)
    fake_position_ids = fake_position_ids.to(args.device)

    with torch.no_grad():
        outputs = model(
            input_ids=fake_input_ids,
            attention_mask=fake_attention_mask,
            moe_token_types=fake_moe_token_types,
            position_ids=fake_position_ids,
            proprioception=fake_proprioception,
            agent_pos_mask=fake_agent_pos_mask,
            dof_mask=fake_dof_mask,
            dataset_names=fake_dataset_names,
            mode="validate",
        )

    print("Fake inference test successful")
    print(f"Output logits shape: {outputs.logits.shape}")
    print(f"Output logits dtype: {outputs.logits.dtype}")
    print(f"Output logits device: {outputs.logits.device}")
    print(f"Output contains NaN: {torch.isnan(outputs.logits).any().item()}")
    print(f"Output contains infinity: {torch.isinf(outputs.logits).any().item()}")
    print("Output logits statistics:")
    print(f"  Min value: {outputs.logits.min().item():.4f}")
    print(f"  Max value: {outputs.logits.max().item():.4f}")
    print(f"  Mean: {outputs.logits.mean().item():.4f}")
    print(f"  Standard deviation: {outputs.logits.std().item():.4f}")


if __name__ == "__main__":
    main()
