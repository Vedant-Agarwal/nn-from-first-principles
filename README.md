nn-from-first-principles

A complete, from-first-principles walkthrough of neural networks — starting from the simplest possible model (a single neuron learning Celsius → Fahrenheit) and building up step-by-step to a fully working mini Transformer (GPT-style).

This repository contains 11 hands-on Google Colab notebooks that explain every concept clearly, both mathematically and programmatically, using only NumPy before introducing modern deep learning components.

No frameworks.
No magic.
Just the true mechanics of neural networks.

Table of Contents

About This Project

Notebook Series (11 Parts)

Repository Structure

How to Use

Requirements

License

About This Project

nn-from-first-principles is a guided journey through the foundations of deep learning:

understanding neurons

deriving gradients by hand

implementing backpropagation

building your own autograd engine

training multi-layer perceptrons

processing text with embeddings

implementing self-attention

constructing Transformer blocks

generating text autoregressively

By the end, you'll understand exactly how modern neural networks work under the hood, and you’ll have implemented all core components yourself.

🔬 Notebook Series (11 Parts)

Each notebook builds on the last.
Colab links can be added as you upload them.

1. Single Neuron — Celsius → Fahrenheit: Fundamentals: linear model, MSE loss, gradients, gradient descent.

2. Multi-Neuron Networks & Activation Functions: Why we need hidden layers, ReLU, tanh, sigmoid.

3. Forward Pass as Matrix Multiplication: Vectorized operations, batched inputs, GPU-friendly math.

4. Backpropagation From Scratch: Full derivation & implementation of backprop for multi-layer networks.

5. Building an Autograd Engine: Implement a minimal PyTorch-like autodiff system.

6. Training Loops & Optimizers: SGD, Momentum, Adam, initialization strategies, LR schedules.

7. MNIST MLP (From Scratch): Build and train a multi-layer perceptron (without deep learning frameworks).

8. Tokenization & Embeddings: Convert text to numbers, build word & positional embeddings.

9. Self-Attention: Implement Q/K/V, scaled dot-product attention, multi-head attention.

10. Transformer Block: LayerNorm, residual connections, feed-forward layers, attention stack.

11. Mini GPT — Text Generation: Train a tiny GPT-like model on Shakespeare and generate sequences.

Repository Structure
nn-from-first-principles/
│
├── notebooks/
│   ├── 01_single_neuron.ipynb
│   ├── 02_multi_neuron.ipynb
│   ├── 03_forward_pass_matrix.ipynb
│   ├── 04_backprop_from_scratch.ipynb
│   ├── 05_autograd_engine.ipynb
│   ├── 06_training_loops_optimizers.ipynb
│   ├── 07_mnist_mlp.ipynb
│   ├── 08_tokenization_embeddings.ipynb
│   ├── 09_self_attention.ipynb
│   ├── 10_transformer_block.ipynb
│   ├── 11_mini_gpt.ipynb
│
├── assets/
│   ├── diagrams/
│   ├── plots/
│   └── samples/
│
├── extras/
│   ├── distributed_training.ipynb
│   ├── nn_summary_25_pages.ipynb
│
├── README.md
├── LICENSE
└── requirements.txt

For Transformer notebooks:

No external deep learning frameworks required (everything is written manually)
