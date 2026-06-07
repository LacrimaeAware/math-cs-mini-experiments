"""
logistic_regression_animation.py — animate logistic-regression gradient descent.

Builds two Gaussian blobs (classes 0 and 1), trains a logistic-regression
classifier by batch gradient descent, and animates the moving decision boundary
across the training steps. Saves the result to
outputs/linear_classifier_training.mp4.

Requires an ffmpeg binary: set the FFMPEG_PATH environment variable, or have
"ffmpeg" on your PATH (matplotlib will pick it up automatically).
"""

import os
import shutil
import random
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Project-local shared output directory (see prime_lib.py at repo root).
import sys
import pathlib
sys.path.append(next(str(p) for p in pathlib.Path(__file__).resolve().parents if (p / "prime_lib.py").exists()))
from prime_lib import ensure_output_dir

# Use ffmpeg from FFMPEG_PATH or PATH; otherwise fall back to matplotlib's default.
_ffmpeg = os.environ.get("FFMPEG_PATH") or shutil.which("ffmpeg")
if _ffmpeg:
    mpl.rcParams["animation.ffmpeg_path"] = _ffmpeg

random.seed(0)

def gaussian_point(mx, my, s):
    return [random.gauss(mx, s), random.gauss(my, s)]

n = 250
s = 0.4

X0 = [gaussian_point(-1.0, -1.0, s) for _ in range(n)]
X1 = [gaussian_point( 1.0,  1.0, s) for _ in range(n)]

X = X0 + X1
y = [0] * n + [1] * n

X = [[max(-2, min(2, x)), max(-2, min(2, y))] for x, y in X]

Xb = [[1.0, x, y] for x, y in X]

def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))

def loss_and_grad(w):
    loss = 0.0
    g0 = g1 = g2 = 0.0
    m = len(Xb)
    eps = 1e-12

    for i in range(m):
        z = w[0]*Xb[i][0] + w[1]*Xb[i][1] + w[2]*Xb[i][2]
        p = sigmoid(z)
        loss += -(y[i]*math.log(p+eps) + (1-y[i])*math.log(1-p+eps))
        d = p - y[i]
        g0 += d * Xb[i][0]
        g1 += d * Xb[i][1]
        g2 += d * Xb[i][2]

    return loss/m, [g0/m, g1/m, g2/m]

steps = 180
lr = 0.9

m0, b0 = 0.5, 0.0
w = [b0, -m0, 1.0]

w_hist = []
loss_hist = []

for _ in range(steps):
    L, g = loss_and_grad(w)
    w_hist.append(w[:])
    loss_hist.append(L)
    w[0] -= lr * g[0]
    w[1] -= lr * g[1]
    w[2] -= lr * g[2]

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter([p[0] for p in X0], [p[1] for p in X0], alpha=0.8)
ax.scatter([p[0] for p in X1], [p[1] for p in X1], alpha=0.8)
ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-2.2, 2.2)
ax.grid(True, alpha=0.3)

(line,) = ax.plot([], [], lw=2)
text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")

xs = [-2.2, 2.2]

def w_to_mb(w):
    m = -w[1] / w[2]
    b = -w[0] / w[2]
    return m, b

def update(i):
    m, b = w_to_mb(w_hist[i])
    ys = [m*x + b for x in xs]
    line.set_data(xs, ys)
    text.set_text(f"step {i+1}  loss {loss_hist[i]:.4f}")
    return line, text

ani = FuncAnimation(fig, update, frames=steps, interval=30, blit=True)

writer = FFMpegWriter(fps=30, bitrate=1800)
ani.save(str(ensure_output_dir() / "linear_classifier_training.mp4"), writer=writer)
