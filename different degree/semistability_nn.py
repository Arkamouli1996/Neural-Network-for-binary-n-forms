r"""
Semistability classifier for homogeneous polynomials in two variables (degrees 2..8)

Overview
- Task: Classify a homogeneous polynomial f(x, y) of degree d (2 ≤ d ≤ 8) as semistable vs unstable using the GIT
  multiplicity criterion on P^1: semistable iff all roots have multiplicity ≤ d/2; unstable iff some root has
  multiplicity > d/2.
- Input to NN (27-D): Complex coefficients in the standard basis [x^d, x^{d-1}y, ..., y^d] are L2-normalized in C^{d+1},
  left-padded with zeros to length 9, then embedded as a 27-dimensional real vector by concatenating:
  • Re(...) (length 9), then Im(...) (length 9), then a 9-D support mask M that marks active coefficient slots
    (last d+1 positions are 1, left-pad positions are 0). The mask disambiguates padding zeros from real zeros and
    encodes degree implicitly.
- Output: One-hot vector: (1, 0) for semistable, (0, 1) for unstable.

Data generation
- Exactly 10,000 samples per degree d ∈ {2..8} (total 70,000):
  • Include all monomials x^i y^{d-i}, i=0..d (d+1 samples) for each degree.
    • Fill the remaining for that degree with 50% generic complex polynomials (semistable almost surely) and 50%
        certifiably unstable polynomials by enforcing a multiplicity m > d/2 at a linear factor (a x + b y)^m.
        We sample m uniformly from {floor(d/2)+1, ..., d}. The direction (a,b) is random, with a small probability
        p_axis≈0.2 biased to axis-aligned choices (1,0) or (0,1) to include monomial-like cases. The cofactor of
        degree d−m is generic and not divisible by L. Semistable labels are
    verified by a deterministic multiplicity check; unstable samples are constructed by design.

Model
- 3-layer fully connected MLP with 1 hidden layer of 2048 dimensions and ReLU activation, trained in NumPy with full-batch cross-entropy and Adam (β1=0.9, β2=0.999).

Deterministic instability check and graphs
- A deterministic multiplicity checker (univariate reduction + root clustering) verifies instability (max multiplicity > d/2).
- After training, figures are saved under ./plots with timestamped filenames:
  • PCA scatter of the training data (2D projection of the 27-D embedding: 18 Re/Im + 9 mask).
  • Loss and accuracy curves (train/validation) over epochs.
  • A focused curve for the monomial x^4 over epochs.

Post-training UI
- After training completes, a small UI opens to let you:
  • Select degree d ∈ {2..8}.
  • Enter coefficients in the standard basis using complex literals a+bj (a is treated as a+0j).
  • View the NN probabilities (semistable vs unstable) based on the 27-D embedding (Re/Im + mask), and the deterministic decision.
- To suppress the UI (e.g., headless), set SKIP_UI=1.

How to run (Windows PowerShell)
1) Activate the repo’s venv:  .\myvirtual\Scripts\Activate.ps1
2) Run training:             python .\semistability_nn.py
Optional quick smoke-test:   $env:FAST_TEST = 1; python .\semistability_nn.py
"""
import tkinter as tk
from tkinter import ttk, messagebox
import os
import math
import random
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

RNG_SEED = 42
np.random.seed(RNG_SEED)
random.seed(RNG_SEED)

# ------------------------------
# Data generation utilities
# ------------------------------

def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def pad_to_len9(coeffs: np.ndarray) -> np.ndarray:
    """Pad a degree-d coefficient vector [c_0, ..., c_d] for [x^d, ..., y^d] to length 9.
    We left-pad with zeros so the rightmost (d+1) entries are the coefficients.
    """
    d_plus_1 = coeffs.shape[0]
    if d_plus_1 > 9:
        raise ValueError("Degree > 8 not supported; expected length ≤ 9")
    pad_len = 9 - d_plus_1
    if pad_len > 0:
        coeffs = np.concatenate([np.zeros(pad_len, dtype=coeffs.dtype), coeffs])
    return coeffs


def random_generic_poly_coeffs_complex(d: int) -> np.ndarray:
    """Random complex coefficients for a degree-d homogeneous polynomial.
    Basis order: [x^d, x^{d-1}y, ..., y^d]."""
    real = np.random.normal(loc=0.0, scale=1.0, size=(d + 1,))
    imag = np.random.normal(loc=0.0, scale=1.0, size=(d + 1,))
    coeffs = real + 1j * imag
    # Avoid all-zero vector; ensure not tiny
    if np.linalg.norm(coeffs) < 1e-10:
        coeffs[0] = 1.0 + 0.0j
    return coeffs


def _poly_eval_t_chart(coeffs: np.ndarray, t: complex) -> complex:
    """Evaluate on chart y=1: p(t)=c0 t^n + c1 t^{n-1} + ... + c_n."""
    n = coeffs.shape[0] - 1
    val = coeffs[0]
    for i in range(1, n + 1):
        val = val * t + coeffs[i]
    return val


def _poly_eval_s_chart(coeffs: np.ndarray, s: complex) -> complex:
    """Evaluate on chart x=1: q(s)=c0 + c1 s + ... + c_n s^n."""
    n = coeffs.shape[0] - 1
    val = coeffs[n]
    for i in range(n - 1, -1, -1):
        val = val * s + coeffs[i]
    return val


def _convolve_binary_forms(c1: np.ndarray, c2: np.ndarray) -> np.ndarray:
    """Multiply two binary forms given in [x^n, x^{n-1}y, ..., y^n] ordering (1D conv)."""
    n1 = c1.shape[0] - 1
    n2 = c2.shape[0] - 1
    out = np.zeros(n1 + n2 + 1, dtype=np.complex128)
    # out[j] corresponds to y^j
    for j in range(n1 + n2 + 1):
        s = 0+0j
        i_min = max(0, j - n2)
        i_max = min(n1, j)
        for i in range(i_min, i_max + 1):
            s += c1[i] * c2[j - i]
        out[j] = s
    return out


def unstable_poly_coeffs_complex(d: int, tol: float = 1e-10, p_axis: float = 0.2) -> np.ndarray:
    """Construct an unstable polynomial by enforcing a repeated linear factor (a x + b y)^m with m > d/2.
    m is sampled uniformly from {floor(d/2)+1, ..., d}.
    Steps:
      1) Choose (a,b): axis-aligned with probability p_axis (x or y), else random complex unit direction.
      2) Sample m ∈ {floor(d/2)+1, ..., d}.
      3) Form L(x,y) = a x + b y and L^m with coefficients l[i] = C(m,i) a^{m-i} b^{i} in [x^m, ..., y^m] order.
      4) Draw a generic complex cofactor g of degree d-m that is NOT divisible by L (checked on appropriate chart).
      5) Return f = L^m * g via binary form convolution.
    """
    # sample multiplicity across full allowed unstable range
    m_min = (d // 2) + 1
    m = random.randint(m_min, d)

    # Choose (a,b): with probability p_axis, bias toward axes to include monomial-like cases.
    if random.random() < p_axis:
        if random.random() < 0.5:
            # Near x-axis: L ≈ x
            a, b = 1.0 + 0j, 0.0 + 0j
        else:
            # Near y-axis: L ≈ y
            a, b = 0.0 + 0j, 1.0 + 0j
    else:
        # random (a,b) with mild anti-degeneracy guard
        for _ in range(100):
            a = np.random.normal() + 1j * np.random.normal()
            b = np.random.normal() + 1j * np.random.normal()
            if abs(a) < tol and abs(b) < tol:
                continue
            # normalize
            norm = math.sqrt((abs(a)**2) + (abs(b)**2))
            a /= norm
            b /= norm
            # avoid extreme degeneracy so numerical checks remain stable
            if min(abs(a), abs(b)) >= 1e-2:
                break
        else:
            a, b = 1.0 + 0j, 0.1 + 0j

    # L^m coefficients
    l = np.zeros(m + 1, dtype=np.complex128)
    for i in range(m + 1):
        l[i] = (math.comb(m, i)) * (a ** (m - i)) * (b ** i)

    # cofactor g of degree n=d-m, ensure not divisible by L
    n = d - m
    if n == 0:
        g = np.array([1.0 + 0.0j], dtype=np.complex128)
    else:
        for _ in range(200):
            g = random_generic_poly_coeffs_complex(n)
            if abs(a) > tol:
                t0 = -b / a
                val = _poly_eval_t_chart(g, t0)
                if abs(val) > 1e-8:
                    break
            else:
                s0 = -a / b  # here a≈0 => s0≈0
                val = _poly_eval_s_chart(g, s0)
                if abs(val) > 1e-8:
                    break
        else:
            # Fallback: force last coeff to avoid trivial divisibility
            if abs(a) > tol:
                g[-1] += 1.0
            else:
                g[0] += 1.0

    # Convolution to multiply binary forms
    f_coeffs = _convolve_binary_forms(l, g)
    assert f_coeffs.shape[0] == d + 1
    return f_coeffs


def coeffs_complex_to_input18(coeffs: np.ndarray) -> np.ndarray:
    """Map complex coeffs [c0..cd] to 18-d real model input:
    1) L2-normalize in C^{d+1}
    2) Left-pad complex vector to length 9
    3) Concatenate real parts (len 9) followed by imaginary parts (len 9)
    """
    assert np.iscomplexobj(coeffs), "Expected complex coefficients"
    coeffs = l2_normalize(coeffs.astype(np.complex128))
    padded = pad_to_len9(coeffs)
    real = np.real(padded).astype(np.float64)
    imag = np.imag(padded).astype(np.float64)
    return np.concatenate([real, imag], axis=0)


def coeffs_complex_to_input27_mask(coeffs: np.ndarray) -> np.ndarray:
    """Map complex coeffs [c0..cd] to 27-d real model input:
    1) L2-normalize in C^{d+1}
    2) Left-pad complex vector to length 9
    3) Concatenate [Re(9), Im(9), Mask(9)] where Mask marks active coeff slots (last d+1 are 1, left-pad are 0)
    """
    assert np.iscomplexobj(coeffs), "Expected complex coefficients"
    d = coeffs.shape[0] - 1
    coeffs = l2_normalize(coeffs.astype(np.complex128))
    padded = pad_to_len9(coeffs)
    real = np.real(padded).astype(np.float64)
    imag = np.imag(padded).astype(np.float64)
    mask = np.zeros(9, dtype=np.float64)
    mask[-(d + 1):] = 1.0
    return np.concatenate([real, imag, mask], axis=0)


def max_root_multiplicity_homog(coeffs: np.ndarray, tol: float = 1e-6) -> int:
    """Return the maximum multiplicity of a root on P^1 for a homogeneous binary form.

    coeffs: array [c_0, ..., c_d] for basis [x^d, x^{d-1}y, ..., y^d]
    Strategy:
      - Consider p(t) = c_0 t^d + c_1 t^{d-1} + ... + c_d (finite points [t:1]).
      - Roots at infinity correspond to s=0 of q(s) = c_0 + c_1 s + ... + c_d s^d;
        multiplicity is the index of first nonzero c_j from the left.
      - For finite roots, we compute complex roots via numpy.roots, then cluster by proximity
        with a fixed tolerance 'tol' to infer multiplicities.
    """
    d = coeffs.shape[0] - 1
    # Multiplicity at infinity: number of leading zeros from c_0 forward
    m_inf = 0
    for j in range(d + 1):
        if abs(coeffs[j]) <= tol:
            m_inf += 1
        else:
            break
    # Handle all-zero edge case defensively (shouldn't occur after normalization)
    if m_inf == d + 1:
        return d + 1

    # Finite roots via univariate polynomial p(t)
    # np.roots expects coefficients from highest power to constant
    p_coeffs = coeffs.astype(np.complex128)
    # If leading coefficients are ~0, np.roots will effectively reduce degree
    roots = np.roots(p_coeffs)
    if roots.size == 0:
        return max(m_inf, 0)

    # Cluster roots within tolerance (hierarchical single-link style)
    # Deterministic ordering via sort by real, then imag
    order = np.lexsort((roots.imag, roots.real))
    roots_sorted = roots[order]
    clusters: List[List[complex]] = []
    for r in roots_sorted:
        if not clusters:
            clusters.append([r])
            continue
        # assign to latest cluster if close to last member
        if abs(r - clusters[-1][-1]) <= tol:
            clusters[-1].append(r)
        else:
            clusters.append([r])

    max_mult_finite = max(len(c) for c in clusters) if clusters else 0
    return max(m_inf, max_mult_finite)


def is_unstable_homog(coeffs: np.ndarray, tol: float = 1e-6) -> bool:
    """Deterministically decide instability: unstable iff max multiplicity > d/2."""
    d = coeffs.shape[0] - 1
    m = max_root_multiplicity_homog(coeffs, tol=tol)
    return m > d / 2.0


def generate_dataset(per_degree_n: int = 10_000,
                     degrees: List[int] | None = None,
                     include_all_monomials: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Generate dataset (X, Y) with 10,000 samples per degree d ∈ {2..8}.
    - X: shape (N, 27) real-valued vectors built from complex coefficients (Re, Im, 9-D support mask)
    - Y: shape (N, 2) one-hot labels: [1,0]=semistable, [0,1]=unstable
    N = sum_d per_degree_n, default N = 7 * 10,000 = 70,000.
    """
    if degrees is None:
        degrees = list(range(2, 9))  # 2..8 inclusive

    samples: List[np.ndarray] = []
    labels: List[np.ndarray] = []

    for d in degrees:
        count_added = 0
        # 1) Include all monomials for degree d
        if include_all_monomials:
            for i in range(d + 1):
                coeffs = np.zeros(d + 1, dtype=np.complex128)
                coeffs[i] = 1.0 + 0.0j  # x^{d-i} y^{i}
                x = coeffs_complex_to_input27_mask(coeffs)
                label_unstable = is_unstable_homog(coeffs)
                y = np.array([0.0, 1.0], dtype=np.float64) if label_unstable else np.array([1.0, 0.0], dtype=np.float64)
                samples.append(x)
                labels.append(y)
                count_added += 1

        # 2) Fill the remaining for this degree: 50% generic (semistable), 50% unstable
        remaining = max(0, per_degree_n - count_added)
        # Split remainder; if odd, let semistable get the extra by ceiling
        n_semistable = remaining - remaining // 2
        n_unstable = remaining // 2

        # Semistable/generic for degree d
        for _ in range(n_semistable):
            # Ensure semistable under deterministic check (resample if not)
            attempts = 0
            while True:
                coeffs = random_generic_poly_coeffs_complex(d)
                if not is_unstable_homog(coeffs):
                    break
                attempts += 1
                if attempts > 50:
                    # fallback: perturb to break repeated factors
                    coeffs = coeffs + 1e-3 * (np.random.randn(d + 1) + 1j*np.random.randn(d + 1))
                    if not is_unstable_homog(coeffs):
                        break
            x = coeffs_complex_to_input27_mask(coeffs)
            y = np.array([1.0, 0.0], dtype=np.float64)  # semistable
            samples.append(x)
            labels.append(y)

        # Unstable for degree d
        for _ in range(n_unstable):
            # Construct unstable polynomial by design using random linear factor (ax+by)^m
            coeffs = unstable_poly_coeffs_complex(d)
            x = coeffs_complex_to_input27_mask(coeffs)
            y = np.array([0.0, 1.0], dtype=np.float64)  # unstable
            samples.append(x)
            labels.append(y)

    X = np.vstack(samples)
    Y = np.vstack(labels)

    # Shuffle
    idx = np.random.permutation(X.shape[0])
    X = X[idx]
    Y = Y[idx]
    return X, Y


# ------------------------------
# Simple NumPy MLP (1 hidden layer with 2048 dimensions)
# ------------------------------

@dataclass
class MLPConfig:
    input_dim: int = 27
    hidden_dims: Tuple[int, ...] = (2048,)
    output_dim: int = 2
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8


class MLP:
    def __init__(self, cfg: MLPConfig):
        self.cfg = cfg
        layer_dims = [cfg.input_dim] + list(cfg.hidden_dims) + [cfg.output_dim]
        self.W: List[np.ndarray] = []
        self.b: List[np.ndarray] = []
        # Adam state
        self.mW: List[np.ndarray] = []
        self.vW: List[np.ndarray] = []
        self.mb: List[np.ndarray] = []
        self.vb: List[np.ndarray] = []
        self.t: int = 0
        for i in range(len(layer_dims) - 1):
            fan_in = layer_dims[i]
            fan_out = layer_dims[i + 1]
            w = np.random.randn(fan_in, fan_out) * math.sqrt(2.0 / fan_in)
            b = np.zeros((1, fan_out))
            self.W.append(w)
            self.b.append(b)
            # init Adam buffers
            self.mW.append(np.zeros_like(w))
            self.vW.append(np.zeros_like(w))
            self.mb.append(np.zeros_like(b))
            self.vb.append(np.zeros_like(b))

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    @staticmethod
    def relu_grad(x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(np.float64)

    @staticmethod
    def softmax(logits: np.ndarray) -> np.ndarray:
        z = logits - np.max(logits, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Returns (pre-activations, activations). activations[0] = X"""
        a = X
        pre_acts: List[np.ndarray] = []
        activations: List[np.ndarray] = [X]
        for i in range(len(self.W) - 1):
            z = a @ self.W[i] + self.b[i]
            pre_acts.append(z)
            a = self.relu(z)
            activations.append(a)
        # Last layer (logits)
        zL = activations[-1] @ self.W[-1] + self.b[-1]
        pre_acts.append(zL)
        activations.append(zL)
        return pre_acts, activations

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        _, activations = self.forward(X)
        logits = activations[-1]
        return self.softmax(logits)

    def penultimate(self, X: np.ndarray) -> np.ndarray:
        """Return activations of the last hidden layer (penultimate representation)."""
        _, activations = self.forward(X)
        return activations[-2]

    def loss_and_grads(self, X: np.ndarray, Y: np.ndarray) -> Tuple[float, List[np.ndarray], List[np.ndarray]]:
        pre, acts = self.forward(X)
        logits = acts[-1]
        probs = self.softmax(logits)
        # Cross-entropy
        eps = 1e-12
        loss = -np.mean(np.sum(Y * np.log(probs + eps), axis=1))

        # Backprop
        grads_W: List[np.ndarray] = [None] * len(self.W)
        grads_b: List[np.ndarray] = [None] * len(self.b)

        # dL/dlogits = probs - Y (for softmax + CE)
        delta = (probs - Y) / X.shape[0]

        # Last layer grads
        grads_W[-1] = acts[-2].T @ delta
        grads_b[-1] = np.sum(delta, axis=0, keepdims=True)

        # Hidden layers (reverse)
        for i in range(len(self.W) - 2, -1, -1):
            delta = (delta @ self.W[i + 1].T) * self.relu_grad(pre[i])
            grads_W[i] = acts[i].T @ delta
            grads_b[i] = np.sum(delta, axis=0, keepdims=True)

        return loss, grads_W, grads_b

    def step(self, grads_W: List[np.ndarray], grads_b: List[np.ndarray], lr: float | None = None) -> None:
        # Adam optimizer
        if lr is None:
            lr = self.cfg.lr
        b1, b2, eps = self.cfg.beta1, self.cfg.beta2, self.cfg.eps
        self.t += 1
        for i in range(len(self.W)):
            gW = grads_W[i]
            gB = grads_b[i]

            self.mW[i] = b1 * self.mW[i] + (1 - b1) * gW
            self.vW[i] = b2 * self.vW[i] + (1 - b2) * (gW * gW)
            self.mb[i] = b1 * self.mb[i] + (1 - b1) * gB
            self.vb[i] = b2 * self.vb[i] + (1 - b2) * (gB * gB)

            mW_hat = self.mW[i] / (1 - (b1 ** self.t))
            vW_hat = self.vW[i] / (1 - (b2 ** self.t))
            mb_hat = self.mb[i] / (1 - (b1 ** self.t))
            vb_hat = self.vb[i] / (1 - (b2 ** self.t))

            self.W[i] -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
            self.b[i] -= lr * mb_hat / (np.sqrt(vb_hat) + eps)


# ------------------------------
# Train / evaluate
# ------------------------------

def accuracy(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    y_pred = np.argmax(y_proba, axis=1)
    y_t = np.argmax(y_true, axis=1)
    return float(np.mean(y_pred == y_t))


def _infer_degree_from_mask(X: np.ndarray) -> np.ndarray:
    """Infer degree d from the 9-D mask in our 27-D input: last d+1 entries are 1s.
    X shape: (N,27). Returns int array of shape (N,).
    """
    mask = X[:, 18:27]
    d = np.sum(mask, axis=1).astype(int) - 1
    return d


def _degree_to_mask_vec(d: int) -> np.ndarray:
    m = np.zeros(9, dtype=np.float64)
    m[-(d + 1):] = 1.0
    return m


def _train_linear_probe(features: np.ndarray, degrees: np.ndarray, reg: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
    """Train a simple ridge-regularized linear classifier to predict degree index (2..8 -> 0..6).
    Returns (W, b) where logits = F @ W + b.
    """
    # Map degrees {2..8} -> {0..6}
    y_idx = degrees - 2
    num_classes = 7
    Y = np.eye(num_classes, dtype=np.float64)[y_idx]
    F = features.astype(np.float64)
    N, D = F.shape
    # Add bias via closed-form ridge: solve for W_ext with augmented design
    F_ext = np.hstack([F, np.ones((N, 1), dtype=np.float64)])
    I = np.eye(D + 1, dtype=np.float64)
    I[-1, -1] = 0.0  # don't regularize bias
    W_ext = np.linalg.pinv(F_ext.T @ F_ext + reg * I) @ F_ext.T @ Y
    W = W_ext[:-1, :]
    b = W_ext[-1:, :]
    return W, b


def _probe_accuracy(features: np.ndarray, degrees: np.ndarray, W: np.ndarray, b: np.ndarray) -> float:
    logits = features @ W + b
    preds = np.argmax(logits, axis=1)
    truth = degrees - 2
    return float(np.mean(preds == truth))


def evaluate_degree_from_mask_and_features(model: MLP, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[float, float]:
    """Return (linear-probe train acc, test acc) for predicting degree from penultimate features."""
    d_train = _infer_degree_from_mask(X_train)
    d_test = _infer_degree_from_mask(X_test)
    F_train = model.penultimate(X_train)
    F_test = model.penultimate(X_test)
    W, b = _train_linear_probe(F_train, d_train)
    return _probe_accuracy(F_train, d_train, W, b), _probe_accuracy(F_test, d_test, W, b)


def evaluate_mask_ablation(model: MLP, X_test: np.ndarray, Y_test: np.ndarray) -> dict:
    """Evaluate how predictions change when the 9-D mask is zeroed, shuffled, or set to random degrees."""
    out = {}
    # Baseline
    out['baseline_acc'] = accuracy(Y_test, model.predict_proba(X_test))
    # Zero mask
    X_zero = X_test.copy()
    X_zero[:, 18:27] = 0.0
    out['zero_mask_acc'] = accuracy(Y_test, model.predict_proba(X_zero))
    # Shuffled mask across samples
    X_shuf = X_test.copy()
    perm = np.random.permutation(X_test.shape[0])
    X_shuf[:, 18:27] = X_test[perm, 18:27]
    out['shuffled_mask_acc'] = accuracy(Y_test, model.predict_proba(X_shuf))
    # Random degree mask per sample
    X_rand = X_test.copy()
    rand_deg = np.random.randint(2, 9, size=(X_test.shape[0],))
    masks = np.vstack([_degree_to_mask_vec(int(d)) for d in rand_deg])
    X_rand[:, 18:27] = masks
    out['random_degree_mask_acc'] = accuracy(Y_test, model.predict_proba(X_rand))
    return out


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def plot_training_curves(loss_hist: List[float],
                         train_acc_hist: List[float],
                         test_acc_hist: List[float],
                         out_dir: str,
                         title_override: str | None = None) -> str:
    os.makedirs(out_dir, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    epochs = np.arange(1, len(loss_hist) + 1)
    # Loss on left axis
    ax1.plot(epochs, loss_hist, color='tab:red', label='Loss (Cross-Entropy)', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.grid(True, linestyle='--', alpha=0.3)

    # Accuracy on right axis
    ax2 = ax1.twinx()
    ax2.plot(epochs, np.array(train_acc_hist) * 100.0, color='tab:blue', label='Train Accuracy (%)', linewidth=2)
    ax2.plot(epochs, np.array(test_acc_hist) * 100.0, color='tab:green', label='Validation Accuracy (%)', linewidth=2)
    ax2.set_ylabel('Accuracy (%)', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    # Legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='lower right')

    plt.suptitle(title_override or 'Training Dynamics: Loss and Accuracy over Epochs')
    caption = (
        'Caption: Optimization trajectory of the 3-layer MLP (1 hidden layer with 2048 dimensions) trained with Adam on the selected degrees. The red curve\n'
        'shows cross-entropy loss per epoch (left axis), while the blue and green curves show training and validation\n'
        'accuracy (right axis). Higher accuracy indicates correct discrimination between semistable (all root\n'
        'multiplicities ≤ d/2) and unstable polynomials (∃ root with multiplicity > d/2). Divergence between train and\n'
        'validation curves suggests overfitting; parallel curves with decreasing loss indicate stable convergence.\n'
        'Values reflect full-batch updates with Adam (β1=0.9, β2=0.999).'
    )
    plt.figtext(0.5, -0.05, caption, wrap=True, ha='center', va='top', fontsize=9)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    out_path = os.path.join(out_dir, f"training_curves_{_timestamp()}.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def plot_sample_construction(per_degree_n: int,
                             degrees: List[int],
                             include_all_monomials: bool,
                             out_dir: str) -> str:
    """Visualize how the dataset is constructed per degree as stacked bars.

    For each degree d, counts are:
      - Monomials: (d+1) if include_all_monomials else 0
      - Remaining: per_degree_n - monomials
      - Generic semistable: remaining - remaining//2
      - Constructed unstable: remaining//2
    """
    os.makedirs(out_dir, exist_ok=True)
    Ds = list(degrees)
    monos = []
    semis = []
    unstables = []
    for d in Ds:
        m = (d + 1) if include_all_monomials else 0
        rem = max(0, per_degree_n - m)
        semi = rem - rem // 2
        un = rem // 2
        monos.append(m)
        semis.append(semi)
        unstables.append(un)

    # Stacked bars
    x = np.arange(len(Ds))
    width = 0.6
    fig, ax = plt.subplots(figsize=(10, 6))
    p1 = ax.bar(x, monos, width, label='Monomials', color='#4C78A8')
    p2 = ax.bar(x, semis, width, bottom=monos, label='Generic semistable', color='#72B7B2')
    bottom2 = (np.array(monos) + np.array(semis)).tolist()
    p3 = ax.bar(x, unstables, width, bottom=bottom2, label='Constructed unstable', color='#F58518')

    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in Ds])
    ax.set_xlabel('Degree d')
    ax.set_ylabel('Samples per degree')
    ax.set_title('Dataset construction per degree')
    ax.legend(loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # Annotate totals on top
    totals = (np.array(monos) + np.array(semis) + np.array(unstables)).astype(int)
    for i, total in enumerate(totals):
        ax.text(x[i], total + per_degree_n*0.01, f"{total}", ha='center', va='bottom', fontsize=9)

    caption = (
        'Caption: For each d ∈ {2..8}, the dataset includes all monomials (x^i y^{d-i}); the remainder splits 50/50\n'
        'between generic semistable and constructed unstable examples. Unstable examples are built as (a x + b y)^m g\n'
        'with m sampled uniformly from {floor(d/2)+1, ..., d}; (a,b) is random but axis-biased with small probability\n'
        '(≈0.2) toward (1,0) or (0,1) to better cover monomial-like cases (e.g., x^m, y^m). Counts reflect construction\n'
        'before shuffling.'
    )
    plt.figtext(0.5, -0.05, caption, wrap=True, ha='center', va='top', fontsize=9)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    out_path = os.path.join(out_dir, f"sample_construction_{_timestamp()}.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def plot_data_pca(X_train: np.ndarray, Y_train: np.ndarray, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    # Mean-center
    Xc = X_train - X_train.mean(axis=0, keepdims=True)
    # PCA via SVD (on the 27-D real embedding: 18 Re/Im + 9-D support mask)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:2].T  # (27,2)
    Z = Xc @ comps    # (n,2)
    labels = np.argmax(Y_train, axis=1)
    semi = labels == 0
    unst = labels == 1

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(Z[semi, 0], Z[semi, 1], s=12, alpha=0.7, c='tab:blue', label='Semistable (label [1,0])')
    ax.scatter(Z[unst, 0], Z[unst, 1], s=12, alpha=0.7, c='tab:orange', label='Unstable (label [0,1])')
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_title('Training Data: 2D PCA Projection')
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.3)

    caption = (
        'Caption: A two-dimensional PCA projection of the 27-dimensional real embedding of complex coefficient vectors\n'
        '(18 features for Re/Im of the zero-padded coeffs + 9-D support mask). Each point represents one training\n'
        'polynomial from a dataset with 10,000 samples per degree d ∈ {2..8}. Colors denote ground-truth labels\n'
        '(blue = semistable, orange = unstable). Clusters/overlaps reflect the geometry of the dataset under a linear\n'
        'projection; perfect separability is not guaranteed in 2D.'
    )
    plt.figtext(0.5, -0.05, caption, wrap=True, ha='center', va='top', fontsize=9)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    out_path = os.path.join(out_dir, f"training_data_pca_{_timestamp()}.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def plot_x4_focus(prob_hist: List[float], acc_hist: List[float], out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    epochs = np.arange(1, len(prob_hist) + 1)
    ax1.plot(epochs, np.array(prob_hist) * 100.0, color='tab:purple', linewidth=2, label='P(model predicts UNSTABLE) [%] for x^4')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Probability (%)', color='tab:purple')
    ax1.tick_params(axis='y', labelcolor='tab:purple')
    ax1.grid(True, linestyle='--', alpha=0.3)

    ax2 = ax1.twinx()
    ax2.step(epochs, acc_hist, where='post', color='tab:gray', linewidth=2, label='Correctness (1=correct, 0=incorrect)')
    ax2.set_ylabel('Correctness (0/1)', color='tab:gray')
    ax2.set_ylim(-0.05, 1.05)
    ax2.tick_params(axis='y', labelcolor='tab:gray')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower right')

    plt.suptitle('Focus: Evolution of Prediction for the Monomial x^4')
    caption = (
        'Caption: This plot tracks the model’s behavior on f(x,y)=x^4 (coeffs [1,0,0,0,0]). The purple curve is the\n'
        'probability assigned to the UNSTABLE class across epochs; the gray step line marks prediction correctness.\n'
        'Ground truth: x^4 has a root of multiplicity 4 at t=0 on the y=1 chart (p(t)=t^4), so 4>d/2 and it is unstable.'
    )
    plt.figtext(0.5, -0.05, caption, wrap=True, ha='center', va='top', fontsize=9)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    out_path = os.path.join(out_dir, f"x4_focus_{_timestamp()}.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def plot_monomial_focus(prob_hist_x: List[float], acc_hist_x: List[int],
                        prob_hist_y: List[float], acc_hist_y: List[int],
                        out_dir: str,
                        title: str) -> str:
    """Overlay evolution of P(UNSTABLE) and correctness for x^4 and y^4 across epochs.

    - prob_hist_* are probabilities in [0,1]
    - acc_hist_* are 0/1 correctness flags
    - Title should state training regime, e.g., 'Training: All degrees' or 'Training: Degree 4 only'.
    """
    os.makedirs(out_dir, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    epochs = np.arange(1, len(prob_hist_x) + 1)

    # Probabilities on left axis
    ax1.plot(epochs, np.array(prob_hist_x) * 100.0, color='tab:purple', linewidth=2, label='x^4: P(UNSTABLE) [%]')
    ax1.plot(epochs, np.array(prob_hist_y) * 100.0, color='tab:orange', linewidth=2, label='y^4: P(UNSTABLE) [%]')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Probability (%)', color='tab:purple')
    ax1.tick_params(axis='y', labelcolor='tab:purple')
    ax1.grid(True, linestyle='--', alpha=0.3)

    # Correctness on right axis
    ax2 = ax1.twinx()
    ax2.step(epochs, acc_hist_x, where='post', color='tab:gray', linewidth=2, label='x^4: Correctness (0/1)')
    ax2.step(epochs, acc_hist_y, where='post', color='tab:green', linewidth=2, label='y^4: Correctness (0/1)')
    ax2.set_ylabel('Correctness (0/1)', color='tab:gray')
    ax2.set_ylim(-0.05, 1.05)
    ax2.tick_params(axis='y', labelcolor='tab:gray')

    # Legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower right')

    plt.suptitle(f'Focus: Evolution for monomials x^4 and y^4 — {title}')
    caption = (
        'Caption: This plot tracks model behavior on the monomials x^4 and y^4. Colored curves show the probability\n'
        'assigned to the UNSTABLE class; step lines indicate correctness per epoch. Titles indicate whether the model\n'
        'was trained on all degrees or only on degree 4. Ground truth: both x^4 and y^4 are unstable since each has a\n'
        'root of multiplicity 4 on an affine chart (4 > d/2 for d=4).'
    )
    plt.figtext(0.5, -0.05, caption, wrap=True, ha='center', va='top', fontsize=9)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    out_path = os.path.join(out_dir, f"monomial_focus_{_timestamp()}.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def _basis_string_for_degree(d: int) -> str:
    terms = [f"x^{d-i}y^{i}" for i in range(d + 1)]
    # Clean up ^1 and ^0 for nicer display
    def clean(term: str) -> str:
        return (
            term.replace("^1", "").replace("x^0", "1").replace("y^0", "")
        ).replace("1", "", 1) if term.startswith("1") else term
    return ", ".join(clean(t) for t in terms)


def plot_unstable_probe(model: 'MLP', degrees: List[int], out_dir: str, test_acc: float) -> str:
    """Generate one random unstable polynomial for each degree and plot model confidence.

    Bars show the model's predicted probability (in %) for the UNSTABLE class for a single random
    unstable polynomial from each degree in `degrees`. Bar color encodes correctness
    (green = predicted unstable, red = predicted semistable). The caption reports overall test accuracy.
    """
    os.makedirs(out_dir, exist_ok=True)

    ds = []
    probs = []
    correct = []
    m_inferred = []
    for d in degrees:
        # draw until the deterministic checker confirms instability (robustness guard)
        for _ in range(100):
            c = unstable_poly_coeffs_complex(d)
            if is_unstable_homog(c):
                break
        x = coeffs_complex_to_input27_mask(c)[None, :]
        proba = model.predict_proba(x)[0]
        p_unstable = float(proba[1])
        pred = int(np.argmax(proba) == 1)
        m_hat = max_root_multiplicity_homog(c)
        ds.append(d)
        probs.append(p_unstable * 100.0)
        correct.append(bool(pred))  # true label is unstable
        m_inferred.append(int(m_hat))

    colors = ["#2ca02c" if ok else "#d62728" for ok in correct]  # green/red
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([str(d) for d in ds], probs, color=colors, alpha=0.85)
    ax.axhline(50.0, color='gray', linestyle='--', linewidth=1, alpha=0.6)
    for i, (d, p, mhat) in enumerate(zip(ds, probs, m_inferred)):
        ax.text(i, p + 1.0, f"{p:.1f}%\nm={mhat}", ha='center', va='bottom', fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_xlabel('Degree d')
    ax.set_ylabel('P(UNSTABLE) [%]')
    ax.set_title(f'Random Unstable Probes by Degree (one per d) — Test acc: {test_acc*100:.2f}%')
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)

    caption = (
        f"Caption: For each degree d ∈ {{2..8}}, one random unstable polynomial was generated by sampling a multiplicity\n"
        f"m uniformly from {{floor(d/2)+1, ..., d}} and a random linear factor (axis-biased with small probability), then\n"
        f"validated deterministically. Bars show the model’s probability assigned to the UNSTABLE class; colors encode\n"
        f"prediction correctness (green = predicted unstable, red = predicted semistable). Each bar is annotated with the\n"
        f"inferred maximum root multiplicity m̂ from a deterministic checker. The horizontal dashed line at 50% indicates\n"
        f"a neutral threshold. Overall test accuracy for this training session: {test_acc*100:.2f}%."
    )
    plt.figtext(0.5, -0.05, caption, wrap=True, ha='center', va='top', fontsize=9)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    out_path = os.path.join(out_dir, f"random_unstable_probe_{_timestamp()}.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def launch_ui(model: 'MLP') -> None:
    """Launch a small Tkinter UI to collect a polynomial and display predictions.
    - Degree selection d ∈ {2..8}
    - Coefficients entry: comma-separated complex numbers (a+bj), in basis [x^d, x^{d-1}y, ..., y^d]
    - Displays NN probability for (semistable, unstable) and deterministic decision
    """
    root = tk.Tk()
    root.title("Semistability Classifier UI")

    container = ttk.Frame(root, padding=12)
    container.grid(row=0, column=0, sticky="nsew")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    # Degree selection
    ttk.Label(container, text="Degree (d):").grid(row=0, column=0, sticky="w")
    deg_var = tk.IntVar(value=4)
    deg_spin = ttk.Spinbox(container, from_=2, to=8, textvariable=deg_var, width=5)
    deg_spin.grid(row=0, column=1, sticky="w")

    # Basis display
    basis_label_var = tk.StringVar()
    def refresh_basis_label(*_):
        d = int(deg_var.get())
        basis_label_var.set(f"Standard basis for degree {d}: [" + _basis_string_for_degree(d) + "]")
    refresh_basis_label()
    deg_var.trace_add('write', refresh_basis_label)

    basis_lbl = ttk.Label(container, textvariable=basis_label_var, wraplength=600, justify="left")
    basis_lbl.grid(row=1, column=0, columnspan=4, sticky="w", pady=(6, 6))

    # Instructions
    instructions = (
        "Enter coefficients as comma-separated complex numbers in Python literal form a+bj.\n"
        "Examples: 1, 2+3j, -0.5j, 4-2j. If you enter just 'a', it's interpreted as a+0j.\n"
        "Order matches the basis above: [x^d, x^{d-1}y, ..., y^d]."
    )
    ttk.Label(container, text=instructions, wraplength=600, justify="left").grid(row=2, column=0, columnspan=4, sticky="w")

    # Coefficients entry
    ttk.Label(container, text="Coefficients:").grid(row=3, column=0, sticky="w", pady=(8, 0))
    coeff_entry = ttk.Entry(container, width=80)
    coeff_entry.grid(row=3, column=1, columnspan=3, sticky="we", pady=(8, 0))
    container.columnconfigure(3, weight=1)

    # Results area
    result_var = tk.StringVar(value="")
    result_lbl = ttk.Label(container, textvariable=result_var, wraplength=600, justify="left", foreground="#333")
    result_lbl.grid(row=5, column=0, columnspan=4, sticky="w", pady=(10, 0))

    def parse_coeffs(text: str, d: int) -> np.ndarray:
        parts = [p.strip() for p in text.strip().split(',') if p.strip()]
        if len(parts) != d + 1:
            raise ValueError(f"Expected {d+1} coefficients, got {len(parts)}")
        vals = []
        for p in parts:
            try:
                vals.append(complex(p))
            except Exception:
                raise ValueError(f"Invalid complex literal: '{p}'. Use forms like 1, 2+3j, -0.5j")
        return np.array(vals, dtype=np.complex128)

    def ui_prepare_nn_input(coeffs_c: np.ndarray, d: int) -> np.ndarray:
        # For the NN trained on complex coefficients (embedded into 27-D real via Re/Im + 9-D mask),
        # build the 27-D input via normalization in C^{d+1}, padding to length 9, stacking Re/Im and appending mask.
        x = coeffs_complex_to_input27_mask(coeffs_c)
        return x[None, :]

    def on_predict():
        d = int(deg_var.get())
        text = coeff_entry.get()
        try:
            cvec = parse_coeffs(text, d)
        except ValueError as e:
            messagebox.showerror("Invalid input", str(e), parent=root)
            return

        # Deterministic rule on full complex coefficients
        det_is_unstable = is_unstable_homog(cvec)
        det_text = "Unstable (max multiplicity > d/2)" if det_is_unstable else "Semistable (all multiplicities ≤ d/2)"

        # NN prediction
        xin = ui_prepare_nn_input(cvec, d)
        proba = model.predict_proba(xin)[0]
        semi_pct = float(proba[0]) * 100.0
        unst_pct = float(proba[1]) * 100.0
        nn_text = f"Neural net prediction: Semistable {semi_pct:.2f}% | Unstable {unst_pct:.2f}%"

        # Display
        result_var.set(nn_text + "\nDeterministic decision: " + det_text)

    predict_btn = ttk.Button(container, text="Predict", command=on_predict)
    predict_btn.grid(row=4, column=0, sticky="w", pady=(10, 0))

    # Pre-fill example for convenience (x^4)
    deg_var.set(4)
    coeff_entry.insert(0, "1, 0, 0, 0, 0")

    root.mainloop()

def train_and_evaluate(epochs: int = 100,
                       per_degree_n: int = 10_000,
                       train_frac: float = 0.75,
                       lr: float = 1e-3,
                       fast: bool = False,
                       degrees: List[int] | None = None) -> None:
    if fast:
        # Smaller quick run for sanity
        per_degree_n = 1000
        epochs = 5
        lr = 1e-3

    if degrees is None:
        degrees = list(range(2, 9))

    # Visualize dataset construction philosophy (clearer than PCA if mask is confusing)
    out_dir = os.path.join("plots")
    p0 = plot_sample_construction(per_degree_n, degrees, include_all_monomials=True, out_dir=out_dir)

    X, Y = generate_dataset(per_degree_n=per_degree_n, degrees=degrees)
    n_train = int(train_frac * X.shape[0])
    X_train, Y_train = X[:n_train], Y[:n_train]
    X_test, Y_test = X[n_train:], Y[n_train:]

    cfg = MLPConfig(lr=lr)
    model = MLP(cfg)

    # Histories for plotting
    loss_hist: List[float] = []
    train_acc_hist: List[float] = []
    test_acc_hist: List[float] = []

    # Track model behavior on monomials x^4 and y^4 using complex embedding
    coeffs_x4 = np.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    coeffs_y4 = np.array([0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex128)
    x4 = coeffs_complex_to_input27_mask(coeffs_x4)[None, :]
    y4 = coeffs_complex_to_input27_mask(coeffs_y4)[None, :]
    x4_prob_hist: List[float] = []  # probability assigned to UNSTABLE class
    x4_acc_hist: List[int] = []     # 1 if predicted unstable, else 0
    y4_prob_hist: List[float] = []
    y4_acc_hist: List[int] = []

    for ep in range(1, epochs + 1):
        loss, gW, gB = model.loss_and_grads(X_train, Y_train)
        model.step(gW, gB)

        # Metrics each epoch
        train_proba = model.predict_proba(X_train)
        test_proba = model.predict_proba(X_test)
        train_acc = accuracy(Y_train, train_proba)
        test_acc = accuracy(Y_test, test_proba)

        loss_hist.append(loss)
        train_acc_hist.append(train_acc)
        test_acc_hist.append(test_acc)

        # Monomial focus: x^4 and y^4
        x4_proba = model.predict_proba(x4)[0, 1]
        x4_pred = int(np.argmax(model.predict_proba(x4)) == 1)
        y4_proba = model.predict_proba(y4)[0, 1]
        y4_pred = int(np.argmax(model.predict_proba(y4)) == 1)
        x4_prob_hist.append(float(x4_proba))
        x4_acc_hist.append(x4_pred)
        y4_prob_hist.append(float(y4_proba))
        y4_acc_hist.append(y4_pred)

        if ep % max(1, epochs // 10) == 0 or ep == 1:
            print(f"Epoch {ep:3d}/{epochs} | loss={loss:.4f} | train_acc={train_acc*100:.1f}% | test_acc={test_acc*100:.1f}%")

    # Final metrics
    train_acc = accuracy(Y_train, model.predict_proba(X_train))
    test_acc = accuracy(Y_test, model.predict_proba(X_test))
    print("\nFinal:")
    print(f"Train accuracy: {train_acc*100:.2f}%")
    print(f"Test  accuracy: {test_acc*100:.2f}%")

    # Save plots
    p1 = plot_data_pca(X_train, Y_train, out_dir)
    p2 = plot_training_curves(loss_hist, train_acc_hist, test_acc_hist, out_dir)
    p3 = plot_monomial_focus(x4_prob_hist, x4_acc_hist, y4_prob_hist, y4_acc_hist, out_dir, title='Training: All degrees')
    p4 = plot_unstable_probe(model, list(range(2, 9)), out_dir, test_acc)
    print(f"Saved plots:\n - {p0}\n - {p1}\n - {p2}\n - {p3}\n - {p4}")

    # Degree detection tests
    probe_train_acc, probe_test_acc = evaluate_degree_from_mask_and_features(model, X_train, X_test)
    print(f"Degree linear-probe acc (penultimate features): train={probe_train_acc*100:.2f}%, test={probe_test_acc*100:.2f}%")
    mask_eval = evaluate_mask_ablation(model, X_test, Y_test)
    print("Mask ablation accuracies:")
    for k, v in mask_eval.items():
        print(f" - {k}: {v*100:.2f}%")

    # Launch interactive UI unless suppressed
    if os.environ.get("SKIP_UI", "0").strip() not in {"1", "true", "True"}:
        try:
            launch_ui(model)
        except Exception as e:
            print(f"UI could not be launched: {e}")

    # Also generate separate figures for degree-4-only training
    p_deg4_curves, p_deg4_x4 = train_and_plot_deg4_curves(epochs=epochs, per_degree_n=per_degree_n, train_frac=train_frac, lr=lr, fast=fast)
    print(f"Saved degree-4-only figures:\n - {p_deg4_curves}\n - {p_deg4_x4}")
def train_and_plot_deg4_curves(epochs: int = 100,
                               per_degree_n: int = 10_000,
                               train_frac: float = 0.75,
                               lr: float = 1e-3,
                               fast: bool = False) -> Tuple[str, str]:
    """Train only on degree 4 polynomials and save:
    - Training/testing accuracy curves
    - Monomial focus (x^4 and y^4 evolution)
    Returns the two saved figure paths.
    """
    if fast:
        per_degree_n = 1000
        epochs = 5
        lr = 1e-3

    X, Y = generate_dataset(per_degree_n=per_degree_n, degrees=[4])
    n_train = int(train_frac * X.shape[0])
    X_train, Y_train = X[:n_train], Y[:n_train]
    X_test, Y_test = X[n_train:], Y[n_train:]

    cfg = MLPConfig(lr=lr)
    model = MLP(cfg)

    loss_hist: List[float] = []
    train_acc_hist: List[float] = []
    test_acc_hist: List[float] = []

    # Track monomials during deg-4-only training
    coeffs_x4 = np.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    coeffs_y4 = np.array([0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex128)
    x4 = coeffs_complex_to_input27_mask(coeffs_x4)[None, :]
    y4 = coeffs_complex_to_input27_mask(coeffs_y4)[None, :]
    x4_prob_hist: List[float] = []
    x4_acc_hist: List[int] = []
    y4_prob_hist: List[float] = []
    y4_acc_hist: List[int] = []

    for ep in range(1, epochs + 1):
        loss, gW, gB = model.loss_and_grads(X_train, Y_train)
        model.step(gW, gB)
        train_proba = model.predict_proba(X_train)
        test_proba = model.predict_proba(X_test)
        train_acc_hist.append(accuracy(Y_train, train_proba))
        test_acc_hist.append(accuracy(Y_test, test_proba))
        loss_hist.append(loss)
        # Monomial focus (deg-4-only)
        x4_proba = model.predict_proba(x4)[0, 1]
        x4_pred = int(np.argmax(model.predict_proba(x4)) == 1)
        y4_proba = model.predict_proba(y4)[0, 1]
        y4_pred = int(np.argmax(model.predict_proba(y4)) == 1)
        x4_prob_hist.append(float(x4_proba))
        x4_acc_hist.append(x4_pred)
        y4_prob_hist.append(float(y4_proba))
        y4_acc_hist.append(y4_pred)

        if ep % max(1, epochs // 10) == 0 or ep == 1:
            print(f"[deg4-only] Epoch {ep:3d}/{epochs} | loss={loss:.4f} | train_acc={train_acc_hist[-1]*100:.1f}% | test_acc={test_acc_hist[-1]*100:.1f}%")

    out_dir = os.path.join("plots")
    path_curves = plot_training_curves(loss_hist, train_acc_hist, test_acc_hist, out_dir, title_override='Training Dynamics (degree 4 only)')
    path_focus = plot_monomial_focus(x4_prob_hist, x4_acc_hist, y4_prob_hist, y4_acc_hist, out_dir, title='Training: Degree 4 only')
    return path_curves, path_focus


if __name__ == "__main__":
    fast = os.environ.get("FAST_TEST", "0").strip() in {"1", "true", "True"}
    # Defaults: 10,000 per degree (2..8) -> 70,000 total
    # Run for 500 epochs by default (FAST_TEST overrides to a small number)
    train_and_evaluate(epochs=500, per_degree_n=10_000, train_frac=0.75, lr=1e-3, fast=fast)
