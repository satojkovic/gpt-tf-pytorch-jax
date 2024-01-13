import torch
import argparse
import sys
import os
from utils import load_encoder_hparams_and_params
from model import GPT2
from tqdm import tqdm
from torchinfo import summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True, help="Input text")
    parser.add_argument(
        "--n_tokens_to_generate",
        default=40,
        type=int,
        help="number of tokens to generate",
    )
    args = parser.parse_args()

    model_size = "124M"
    models_dir = "models"
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)
    print("hparams:", hparams)
    print("Num. of params:", len(params.keys()))

    print("prompt:", args.prompt)
    input_ids = encoder.encode(args.prompt)
    input_text = encoder.decode(input_ids)
    print("input_ids:", input_ids)

    model = GPT2(params, hparams, drop_p=0.1)
    summary(model, input_size=(1, len(input_ids)), dtypes=[torch.long])

    for _ in tqdm(range(args.n_tokens_to_generate), "generating"):
        logits = model(torch.tensor(input_ids).unsqueeze(0))
        next_id = torch.argmax(logits[0][-1], dim=-1)
        input_ids.append(next_id.item())
    print("Input text:\n", input_text)
    print(
        "Generated:\n",
        encoder.decode(input_ids[len(input_ids) - args.n_tokens_to_generate :]),
    )
