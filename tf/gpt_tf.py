import tensorflow as tf
import argparse
from tqdm import tqdm

import sys
import os

from model import GPT2

# import picoGPT
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../picoGPT"))
from picoGPT.utils import load_encoder_hparams_and_params


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
    print("params:", params.keys())
    print("params[blocks]:", params["blocks"][0].keys())
    print("params[blocks][attn]:", params["blocks"][0]["attn"].keys())
    print("params[blocks][ln_1]:", params["blocks"][0]["ln_1"].keys())
    print("params[blocks][ln_2]:", params["blocks"][0]["ln_2"].keys())
    print("params[blocks][mlp]:", params["blocks"][0]["mlp"].keys())

    print("prompt:", args.prompt)
    input_ids = encoder.encode(args.prompt)
    input_text = encoder.decode(input_ids)
    print("input_ids:", input_ids)

    model = GPT2(params, hparams, drop_p=0.1)
    model.build(input_shape=(1, len(input_ids)))
    model.set_pretrained_weights()
    model.summary()

    # Generate next words
    for _ in tqdm(range(args.n_tokens_to_generate), "generating"):
        logits = model(tf.expand_dims(input_ids, axis=0))
        next_id = tf.argmax(tf.squeeze(logits, axis=0)[-1], axis=-1)
        input_ids.append(next_id.numpy().item())
    print("Input text:\n", input_text)
    print(
        "Generated:\n",
        encoder.decode(input_ids[len(input_ids) - args.n_tokens_to_generate :]),
    )
