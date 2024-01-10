import os
import json
import sys
import requests
from tqdm import tqdm
import torch

# import picoGPT
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../picoGPT"))
from encoder import get_encoder


def save_file(file_path, req):
    filename = os.path.basename(file_path)
    with open(file_path, "wb") as f:
        file_size = int(req.headers["content-length"])
        chunk_size = 1000
        with tqdm(
            ncols=100,
            desc="Fetching " + filename,
            total=file_size,
            unit_scale=True,
            unit="b",
        ) as pbar:
            # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
            for chunk in req.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                pbar.update(chunk_size)


# Copy from picoGPT because picoGPT/utils.py import tensorflow
def download_gpt2_files(model_size, model_dir):
    assert model_size in ["124M", "355M", "774M", "1558M"]
    for filename in [
        "checkpoint",
        "encoder.json",
        "hparams.json",
        "model.ckpt.data-00000-of-00001",
        "model.ckpt.index",
        "model.ckpt.meta",
        "vocab.bpe",
    ]:
        url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
        r = requests.get(f"{url}/{model_size}/{filename}", stream=True)
        r.raise_for_status()

        save_file(os.path.join(model_dir, filename), r)


def download_pyth_gpt2_file(pyth_model_path):
    pyth_model_filename = os.path.basename(pyth_model_path)
    base_url = "https://s3.amazonaws.com/models.huggingface.co/bert"
    r = requests.get(f"{base_url}/{pyth_model_filename}", stream=True)
    r.raise_for_status()

    save_file(pyth_model_path, r)


def load_encoder_hparams_and_params(model_size, models_dir):
    # 124M only in pytorch
    assert model_size in ["124M"]

    model_dir = os.path.join(models_dir, model_size)
    if not os.path.exists(model_dir):  # download files if necessary
        os.makedirs(model_dir, exist_ok=True)
        download_gpt2_files(model_size, model_dir)

    pyth_model_path = os.path.join(models_dir, "gpt2-pytorch_model.bin")
    if not os.path.exists(pyth_model_path):
        download_pyth_gpt2_file(pyth_model_path)

    encoder = get_encoder(model_size, models_dir)
    hparams = json.load(open(os.path.join(model_dir, "hparams.json")))
    params = torch.load(pyth_model_path)

    return encoder, hparams, params
