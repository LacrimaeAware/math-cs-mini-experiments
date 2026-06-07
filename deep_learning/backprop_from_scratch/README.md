# Backpropagation from scratch

Manual backpropagation and a small network fit, worked out by hand and then in
code.

## Contents

- Problem 1: one gradient-descent update for a single sigmoid neuron under MSE
  loss, with the forward and backward pass written out.
- Problem 2: one full backpropagation update for all nine weights of a 2-2-1
  sigmoid network.
- Problem 3: fit a binary image with a small multilayer perceptron (Keras),
  learning pixel intensity from normalized (row, column) coordinates.

## Run

```bash
pip install numpy tensorflow matplotlib
jupyter notebook backprop_from_scratch.ipynb
```

The committed notebook has its cell outputs stripped; run it to regenerate them.
