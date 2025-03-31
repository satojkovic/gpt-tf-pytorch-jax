# gpt-tf-pytorch-jax

This repository provides a Python implementation of GPT-2 from scratch, along with the ability to load pre-trained weights provided by OpenAI. This allows you to generate text using the power of GPT-2 without relying on external libraries.

### Features

* GPT-2 implemented from scratch: Understand the inner workings of GPT-2 by exploring its implementation using basic Python libraries like NumPy.
* Load OpenAI pre-trained weights: Utilize the pre-trained weights provided by OpenAI to generate high-quality text without extensive training.
* Command-line interface: Interact with the model easily using a simple command-line interface to input your starting text and generate continuations.

A detailed explanation of the code is posted on Medium.

[Create your own GPT and generate text with OpenAI’s pre-trained parameters](https://medium.com/@satojkovic/create-your-own-gpt-and-generate-text-with-openais-pre-trained-parameters-8d1632d6c92d)

### How to Use
This repository provides a command-line interface to interact with the GPT-2 model. You can generate text by providing a starting prompt and specifying the length of the generated text (default: 40).

There are three frameworks available for running the code: TensorFlow, PyTorch, and JAX. All three frameworks are implemented in the same way, so you can choose the one that best fits your needs. The code is organized into separate directories for each framework.
The main script for each framework is located in the `tf`, `pytorch`, and `jax` directories, respectively. You can run the script from the command line to generate text using the GPT-2 model.

```bash
$ python tf/gpt_tf.py --prompt "Alan Turing theorized that computers would one day become"
```
```bash
$ python pytorch/gpt.py --prompt "Alan Turing theorized that computers would one day become"
```
```bash
$ python jax/gpt.py --prompt "Alan Turing theorized that computers would one day become"
```

Which generates the following output:

```
the most powerful machines on the planet.
The computer is a machine that can perform complex calculations, and it can perform these calculations in a way that is not possible with human hands.
```
