import argparse
import sys
import os
import jax
from clu import parameter_overview
import jax.numpy as jnp

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

    print("prompt:", args.prompt)
    input_ids = encoder.encode(args.prompt)
    input_ids = jnp.expand_dims(jnp.asarray(input_ids), axis=0)
    input_text = encoder.decode(input_ids)
    print("input_ids:", input_ids)

    # Inspect model structure
    key = jax.random.PRNGKey(0)
    gpt2_params = GPT2(params, hparams, drop_p=0.1).init(
        key, input_ids, deterministic=True
    )
    print(jax.tree_map(lambda x: x.shape, gpt2_params))
