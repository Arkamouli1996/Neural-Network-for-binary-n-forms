# different degree — semistability experiments (precise README)

This README documents the exact contents and how to reproduce the experiments in the `different degree` folder of the repository:
https://github.com/Arkamouli1996/Neural-Network-for-binary-n-forms/tree/main/different%20degree

It is written from the code in `semistability_nn.py` (permalink):
https://github.com/Arkamouli1996/Neural-Network-for-binary-n-forms/blob/c1643647770ec41199ee0048d815a54f48d1879a/different%20degree/semistability_nn.py

Contents
- Overview
- 1) What is "semistability" here
- 2) Aim of the neural network
- 3) How the training data is generated (with exact code snippets used)
- 4) Model & hyperparameters (with exact code snippets)
- 5) Final training results (exact numbers from training log)
- 6) Masking: what it is and how it was used (with ablation numbers)
- Running the experiment (exact commands, flags, environment)
- Reproducibility and artifacts
- Notes, caveats and contact

Overview
The experiments train a small MLP (NumPy implementation) to classify homogeneous binary forms (homogeneous polynomials in x,y) of degrees d ∈ {2..8} as semistable vs unstable under the Geometric Invariant Theory multiplicity criterion on P^1:
- Semistable ⇔ every root of the polynomial (on P^1) has multiplicity ≤ d/2.
- Unstable ⇔ there exists a root with multiplicity > d/2.

1) What is semistability here (precise)
- We consider homogeneous polynomials in two variables (binary forms) of degree d:
  f(x,y) = ∑_{i=0}^d c_i x^{d-i} y^i, with complex coefficients c_i.
- Semistability criterion used by the deterministic checker inside the code:
  - Convert to an affine chart (e.g., y=1) and examine root multiplicities.
  - Compute the maximum root multiplicity on P^1 (finite roots via numeric roots + clustering; multiplicity at infinity by counting leading zeros).
  - The sample is declared unstable iff max_multiplicity > d/2.
- Function in code: max_root_multiplicity_homog(...) and is_unstable_homog(...) implement this rule.

2) Aim of the neural network (precise)
- Input: a 27-dimensional real vector embedding of the complex coefficients:
  - Re of padded coefficients (9 dims) || Im of padded coefficients (9 dims) || 9-D support mask (marks which coefficient slots are active).
- Output: softmax over 2 classes:
  - [1,0] => semistable
  - [0,1] => unstable
- Goal: learn to predict the deterministic multiplicity-based decision from the 27-D embedding (i.e., approximate the rule/structure separating polynomials with a high multiplicity root).
- Secondary analyses in the code:
  - Linear-probe on penultimate features to see whether the network encodes degree.
  - Mask ablations to measure reliance on the mask signal.
  - Focused analyses on monomials (e.g., x^4) across training.

3) How the training data is made (exact code snippets)
- The code generates exactly 10,000 samples per degree by default (per_degree_n = 10_000) for degrees 2..8 (70,000 samples total). For each degree d:
  - Include all monomials x^{d-i} y^i (d+1 samples).
  - Fill the remaining slots with 50% generic complex polynomials (resampled until they are deterministically semistable) and 50% constructed unstable polynomials (guaranteed to have a linear factor to multiplicity m > d/2).
- Key snippets (copied verbatim from `semistability_nn.py`) used to construct data:

```python
def unstable_poly_coeffs_complex(d: int, tol: float = 1e-10, p_axis: float = 0.2) -> np.ndarray:
    # sample multiplicity across full allowed unstable range
    m_min = (d // 2) + 1
    m = random.randint(m_min, d)
    # Choose (a,b): axis-biased with prob p_axis
    if random.random() < p_axis:
        if random.random() < 0.5:
            a, b = 1.0 + 0j, 0.0 + 0j
        else:
            a, b = 0.0 + 0j, 1.0 + 0j
    else:
        # random complex unit direction (with mild anti-degeneracy guard)
        for _ in range(100):
            a = np.random.normal() + 1j * np.random.normal()
            b = np.random.normal() + 1j * np.random.normal()
            if abs(a) < tol and abs(b) < tol:
                continue
            norm = math.sqrt((abs(a)**2) + (abs(b)**2))
            a /= norm
            b /= norm
            if min(abs(a), abs(b)) >= 1e-2:
                break
        else:
            a, b = 1.0 + 0j, 0.1 + 0j
    # Build L^m and multiply by random cofactor g of degree d-m
    l = np.zeros(m + 1, dtype=np.complex128)
    for i in range(m + 1):
        l[i] = (math.comb(m, i)) * (a ** (m - i)) * (b ** i)
    n = d - m
    if n == 0:
        g = np.array([1.0 + 0.0j], dtype=np.complex128)
    else:
        for _ in range(200):
            g = random_generic_poly_coeffs_complex(n)
            # ensure g is not divisible by L (checked on appropriate chart)
            if abs(a) > tol:
                t0 = -b / a
                val = _poly_eval_t_chart(g, t0)
                if abs(val) > 1e-8:
                    break
            else:
                s0 = -a / b
                val = _poly_eval_s_chart(g, s0)
                if abs(val) > 1e-8:
                    break
        else:
            if abs(a) > tol:
                g[-1] += 1.0
            else:
                g[0] += 1.0
    f_coeffs = _convolve_binary_forms(l, g)
    assert f_coeffs.shape[0] == d + 1
    return f_coeffs
```

```python
def coeffs_complex_to_input27_mask(coeffs: np.ndarray) -> np.ndarray:
    # 1) L2-normalize in C^{d+1}
    # 2) Left-pad complex vector to length 9
    # 3) Concatenate [Re(9), Im(9), Mask(9)] where Mask marks active coeff slots (last d+1 are 1)
    assert np.iscomplexobj(coeffs), "Expected complex coefficients"
    d = coeffs.shape[0] - 1
    coeffs = l2_normalize(coeffs.astype(np.complex128))
    padded = pad_to_len9(coeffs)
    real = np.real(padded).astype(np.float64)
    imag = np.imag(padded).astype(np.float64)
    mask = np.zeros(9, dtype=np.float64)
    mask[-(d + 1):] = 1.0
    return np.concatenate([real, imag, mask], axis=0)
```

```python
def generate_dataset(per_degree_n: int = 10_000,
                     degrees: List[int] | None = None,
                     include_all_monomials: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    # ... (detailed logic in the file)
    # For each degree d: add monomials, then fill remainder:
    #   n_semistable = remaining - remaining//2  (generic semistable)
    #   n_unstable   = remaining // 2            (constructed unstable)
    # Samples are L2-normalized, padded to length 9, and embedded as 27-D vectors.
```

- The dataset is shuffled after construction. By default RNG_SEED = 42 is used for reproducibility.

4) Model and hyperparameters (exact code snippets)
- Model: NumPy MLP implemented in `semistability_nn.py`. Architecture and training defaults are:

```python
@dataclass
class MLPConfig:
    input_dim: int = 27
    hidden_dims: Tuple[int, ...] = (2048,)   # single hidden layer with 2048 units
    output_dim: int = 2
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
```

- Weight initialization: He-like normal (w ~ N(0, sqrt(2 / fan_in))).
- Activation: ReLU on hidden layer(s), logits on output followed by softmax.
- Optimizer: Full-batch Adam implemented in NumPy with the configured β1/β2 and eps.
- Default training loop parameters (exact defaults from the top-level call):

```python
# Called at module run:
train_and_evaluate(epochs=500, per_degree_n=10_000, train_frac=0.75, lr=1e-3, fast=fast)
```

- Meaning:
  - epochs = 500 (full experiment)
  - per_degree_n = 10_000 (samples per degree → 7 degrees → 70,000 total)
  - train_frac = 0.75 (75% train / 25% test split)
  - lr = 1e-3 (Adam lr)
  - hidden layer: one hidden layer with 2048 units

- Training loop (core epoch step in code):

```python
for ep in range(1, epochs + 1):
    loss, gW, gB = model.loss_and_grads(X_train, Y_train)
    model.step(gW, gB)          # Adam update (full-batch)
    train_proba = model.predict_proba(X_train)
    test_proba = model.predict_proba(X_test)
    train_acc = accuracy(Y_train, train_proba)
    test_acc = accuracy(Y_test, test_proba)
    # histories appended and periodic prints:
    if ep % max(1, epochs // 10) == 0 or ep == 1:
        print(f"Epoch {ep:3d}/{epochs} | loss={loss:.4f} | train_acc={train_acc*100:.1f}% | test_acc={test_acc*100:.1f}%")
```

5) Final training accuracy (exact numbers from training log)
- The training log included with the folder records epoch snapshots and final metrics. Exact reported results from the run in `training log.txt`:

Epoch snapshots (selected):
- Epoch 50/500 | loss=0.6655 | train_acc=61.5% | test_acc=61.6%
- Epoch 100/500 | loss=0.6171 | train_acc=71.5% | test_acc=71.0%
- Epoch 200/500 | loss=0.4891 | train_acc=81.9% | test_acc=80.9%
- Epoch 300/500 | loss=0.3259 | train_acc=90.9% | test_acc=89.3%
- Epoch 400/500 | loss=0.2080 | train_acc=95.6% | test_acc=94.2%
- Epoch 500/500 | loss=0.1473 | train_acc=96.7% | test_acc=95.4%

Final reported metrics (verbatim):

Final:
Train accuracy: 96.70%
Test  accuracy: 95.43%

- The repository saved figures for this run; examples (from the log):
  - plots\sample_construction_20251026-020415.png
  - plots\training_data_pca_20251026-025816.png
  - plots\training_curves_20251026-025818.png
  - plots\monomial_focus_20251026-025819.png
  - plots\random_unstable_probe_20251026-025819.png

- Degree linear-probe accuracy (penultimate features):
```
train=100.00%, test=100.00%
```
  (indicates the penultimate features linearly separate degree labels perfectly for that run).

6) Masking: what it is and how it was used (precise + ablation)
- What the mask is:
  - The 27-D input is built as: [Re(9), Im(9), Mask(9)].
  - The Mask is a 9-dimensional vector with ones in the last (d+1) positions (active coefficients for degree d) and zeros on the left padding. This both disambiguates padded zeros from true zero coefficients and implicitly encodes the polynomial's degree.
- How it was used:
  - The mask is concatenated to the numeric embedding and provided to the network as explicit features.
  - The code contains a mask-ablation evaluation function `evaluate_mask_ablation(...)` that measures the model's accuracy if:
    - the mask is zeroed,
    - the masks are shuffled across examples,
    - masks are replaced by random degree masks.
- Ablation numbers (exact from training log):

```
Mask ablation accuracies:
 - baseline_acc: 95.43%
 - zero_mask_acc: 49.97%
 - shuffled_mask_acc: 85.83%
 - random_degree_mask_acc: 85.78%
```

Interpretation (precise):
- With the correct mask appended (baseline) the test accuracy is 95.43% for this run.
- Zeroing the mask drops accuracy to ~50%, showing the model relies heavily on degree support information encoded by the mask.
- Shuffling masks or randomizing degrees reduces accuracy to ≈85.8% — the network still uses coefficient geometry (Re/Im) to predict some cases, but performance is degraded compared to the correct mask.

Running the experiment (exact commands)
- Run full experiment (default: 500 epochs, 10k samples per degree):
  - On Windows PowerShell (the file includes Windows run notes):
```powershell
# activate virtualenv (example)
.
myvirtual\Scripts\Activate.ps1

# run full training
python .\semistability_nn.py
```

- Quick smoke test:
```powershell
# runs with smaller per_degree_n and fewer epochs
$env:FAST_TEST = 1
python .\semistability_nn.py
```

- Suppress the Tkinter UI at the end (headless mode):
```powershell
$env:SKIP_UI = 1
python .\semistability_nn.py
```

- Inspect CLI help (script defines argparse; run this to see exact flags):
```bash
python semistability_nn.py --help
```

Reproducibility & environment
- The script sets RNG_SEED = 42 for both numpy and random at the top of the file.
- Recommended Python: 3.8+ (code uses type hints, dataclasses and standard numeric stack).
- Minimal Python packages used by the script (inspect top-of-file imports):
  - numpy
  - matplotlib (for plotting)
  - tkinter (optional UI; built-in on many Python installs)
  - standard library: math, random, dataclasses, datetime, typing
- If you want exact pinned versions, run:
```bash
pip freeze > requirements-freeze.txt
# or inspect interpreter env used for the run that produced training log
```

Artifacts saved by the script
- plots/ — many timestamped PNGs:
  - sample_construction_*.png
  - training_data_pca_*.png
  - training_curves_*.png
  - monomial_focus_*.png
  - random_unstable_probe_*.png
- training log (included): `training log.txt` contains epoch prints and final metrics for the run whose numbers appear above.
- No model checkpoint file is saved by the default script (the NumPy model lives in memory; if you want persistent checkpoints add a save routine).

Notes, caveats, and suggestions
- The MLP is implemented in pure NumPy for clarity and reproducibility, using full-batch updates. This is intentionally lightweight but not optimized for GPU/large-scale.
- The deterministic multiplicity checker uses numeric roots and clustering; it is robust for the constructed examples but be aware of numerical instability for highly degenerate coefficients.
- If you require checkpointing, deterministic environment recreation, or reproducible exact byte-for-byte runs on other machines, add:
  - pip freeze of environment
  - explicit saving/loading of model weights (e.g., np.savez for weights and Adam buffers)
  - log the exact command-line used in `training log.txt` (copying the printed run command into that file is recommended)
- Mask importance: the ablation shows the mask is critical to good performance. If you want a version that does not rely on the mask, train without it and consider providing degree as an explicit auxiliary target or using architectures that learn degree from coefficients alone (but expect lower performance unless more complex inductive biases are added).

Contact
- Repository owner: Arkamouli1996 (GitHub handle)
- Email (as in commit metadata): arkapointer@gmail.com

If you want, I can:
- Add a short wrapper to save and load model parameters to disk (np.savez / np.load), or
- Produce a small requirements.txt with suggested pins for numpy and matplotlib based on your current environment (I can generate it from a pip freeze you provide).

This README is directly derived from the program text and the included `training log.txt` for the run that produced the reported final metrics and plots. It intentionally quotes the code and the exact numbers used/reported in order to be as precise and reproducible as possible.
