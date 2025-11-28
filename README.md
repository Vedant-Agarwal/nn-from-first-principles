nn-from-first-principles

A complete, from-first-principles walkthrough of neural networks — starting from the simplest possible model (a single neuron learning Celsius → Fahrenheit) and building up step-by-step to a fully working mini Transformer (GPT-style).

This repository contains 11 hands-on Google Colab notebooks that explain every concept clearly, both mathematically and programmatically, using only NumPy before introducing modern deep learning components.

Table of Contents

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
