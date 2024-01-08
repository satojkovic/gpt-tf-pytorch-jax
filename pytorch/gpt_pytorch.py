import torch
import argparse
import sys
import os
from utils import load_encoder_hparams


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True, help="Input text")
    parser.add_argument(
        "--model_path", required=True, help="Path to gpt2-pytorch_model.bin"
    )
    parser.add_argument(
        "--n_tokens_to_generate",
        default=40,
        type=int,
        help="number of tokens to generate",
    )
    args = parser.parse_args()

    state_dict = torch.load(args.model_path)
    print(f"state_dict: {len(state_dict.keys())} params")

    model_size = "124M"
    models_dir = "models"
    encoder, hparams = load_encoder_hparams(model_size, models_dir)
    print("hparams:", hparams)

    print("prompt:", args.prompt)
    input_ids = encoder.encode(args.prompt)
    input_text = encoder.decode(input_ids)
    print("input_ids:", input_ids)
