# TDA Playbook for a Singular (2,3) Surface in \(\mathbb{P}^4\)
**Executable guide** for a coding agent to analyze a point cloud of ~5000 uniformly sampled points on the singular \((2,3)\) complete intersection (the *triple‑root locus* of binary quartics).

You provide a CSV of homogeneous coordinates with 10 columns per row:  
`(a0,...,a4, b0,...,b4)` is allowed, **or** just 5 columns `(a0,...,a4)`.  
- If there are **10 columns**, treat each row as a complex point \(a+ i\,b\in\mathbb{C}^5\) (we operate in \(\mathbb{R}^{10}\)).  
- If there are **5 columns**, treat it as a real point in \(\mathbb{R}^5\).

We analyze:
1. **Global homology** via persistent homology (Vietoris–Rips / Alpha complex).  
2. **Local homology** & PCA to **detect the singular locus** (the rational normal quartic \(\nu_4(\mathbb{P}^1)\)).  
3. Build **3D mesh visualizations** suitable for inspection (PLY/OBJ).

> **What we expect topologically (guide for interpretation):**  
> The normalization is \(\mathbb{P}^1\times\mathbb{P}^1\).  
> - Over **\(\mathbb{C}\)**: Betti numbers of the complex surface (normalization) are \(b_0=1,\ b_1=0,\ b_2=2,\ b_3=0,\ b_4=1\).  
> - Over **\(\mathbb{R}\)** (real points): \(\mathbb{P}^1(\mathbb{R})\times \mathbb{P}^1(\mathbb{R}) \simeq S^1\times S^1\) (a torus), so typically \(b_0=1,\ b_1=2,\ b_2=1\) **for the real locus** in generic situations.  
> Our sampled set is in projective space; expect the **global H\_0 ≈ 1** and prominent **H\_1 features** consistent with a torus‑like real locus. The **singular curve** should reveal itself via **local homology anomalies** and **PCA dimension drops**.

---

## 0) Environment

```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install numpy pandas scikit-learn umap-learn ripser persim gudhi open3d matplotlib plotly
```

> Optional: `pyvista` for nicer meshes; `keplermapper` for Mapper graphs.

---

## 1) Load & normalize projective points

- Normalize each homogeneous vector to **unit norm**.  
- For projective identification \(x\sim -x\) (on the unit sphere model of \(\mathbb{RP}^{n}\)), use the **antipodal distance** \(d_\pm(x,y)=\min(\|x-y\|,\|x+y\|)\).  
- If your CSV has **10 cols**, we concatenate real & imaginary parts to 10D and do antipodal handling in \(\mathbb{R}^{10}\).

```python
import numpy as np, pandas as pd

CSV_PATH = "sampled_points.csv"  # <-- replace
Xraw = pd.read_csv(CSV_PATH, header=None).values
assert Xraw.shape[1] in (5,10), "Expect 5 or 10 columns"

def normalize_projective_rows(X):
    # normalize each row to unit norm
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms==0] = 1.0
    Y = X / norms
    # canonical representative for antipodal equivalence: force first nonzero entry positive
    for i,row in enumerate(Y):
        j = np.argmax(np.abs(row) > 1e-14)
        if j < row.size and row[j] < 0:
            Y[i] = -row
    return Y

if Xraw.shape[1]==10:
    # complex points => real concatenation
    X = normalize_projective_rows(Xraw)  # in R^10
else:
    X = normalize_projective_rows(Xraw)  # in R^5

print("Points:", X.shape)
```

**Antipodal distance matrix** (optional but recommended for Vietoris–Rips):

```python
from sklearn.metrics import pairwise_distances

def antipodal_distances(Y):
    # d(x,y) = min(||x-y||, ||x+y||)
    D1 = pairwise_distances(Y, Y, metric='euclidean')
    D2 = pairwise_distances(Y, -Y, metric='euclidean')
    return np.minimum(D1, D2)

D = antipodal_distances(X)
```

---

## 2) Global persistent homology (VR + Alpha)

**Vietoris–Rips** with `ripser` (fast, scalable; can feed a precomputed distance).

```python
from ripser import ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt

r = ripser(D, distance_matrix=True, maxdim=2, n_perm=None)  # compute up to H2
dgms = r['dgms']

plt.figure(figsize=(10,4))
plot_diagrams(dgms, show=True)
```

Interpretation:
- Look for **H0**: one long bar → single connected component.  
- **H1**: two robust bars suggest a torus‑like real locus.  
- **H2**: a persistent H2 class may indicate a 2D void (consistent with a torus has one H2 over \(\mathbb{Z}\)).

**Alpha complex** (geometric, often cleaner in low ambient dimension; for 10D might be heavy).

```python
import gudhi as gd

def alpha_persistence(Y):
    ac = gd.AlphaComplex(points=Y.astype(np.float64))
    st = ac.create_simplex_tree()
    st.compute_persistence()
    dgms = st.persistence_intervals_in_dimension
    return st, [dgms(0), dgms(1), dgms(2)]

# For high-dim (R^10), consider subsampling first:
if X.shape[1] <= 6 and X.shape[0] <= 8000:
    st_alpha, dgms_alpha = alpha_persistence(X)
```

> Cross‑check VR vs Alpha: robust features should coincide (H0≈1, H1≈2 prominent for real torus‑like locus). Remember noise + sampling density may blur H2.

---

## 3) Local homology & PCA to detect the **singular locus** \(\nu_4(\mathbb{P}^1)\)

At a **regular** point of a 2D real surface, the **link** (intersection with a small sphere) has the homology of \(S^1\): local H1≈1.  
Along the **singular curve** (1D), the local structure is not a smooth 2D manifold; local homology and PCA change (effective tangent dimension \(\approx 1\) or anisotropic 2D with lower second singular value).

**Algorithm:**  
1. For each point \(x_i\), grab its \(k\)-NN neighborhood (or ε‑ball) with **antipodal metric**.  
2. Run small‑scale VR persistence on the neighborhood to estimate **local Betti numbers**.  
3. Run **PCA** on the neighborhood: record singular values \(\sigma_1\ge\sigma_2\ge\sigma_3\).  
4. Flag points as **singular‑curve candidates** if:
   - Local H1 deviates from ≈1 **or**
   - \(\sigma_2/\sigma_1\) is very small (1D‑like), while density is high.

```python
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

k = 50  # neighborhood size (tune 30–100)
# kNN in antipodal metric:
D = antipodal_distances(X)
nbrs = NearestNeighbors(n_neighbors=k, metric='precomputed').fit(D)
dists, idxs = nbrs.kneighbors(D)

local_scores = np.zeros(X.shape[0])
dim_ratio = np.zeros(X.shape[0])

for i in range(X.shape[0]):
    neigh = X[idxs[i]]
    # local VR persistence up to H1
    Dloc = pairwise_distances(neigh, neigh, metric='euclidean')
    rloc = ripser(Dloc, distance_matrix=True, maxdim=1)
    H0, H1 = rloc['dgms'][0], rloc['dgms'][1]
    # crude score: total persistence in H1 (should be ~1 strong class for regular points)
    pers_H1 = np.sum(H1[:,1]-H1[:,0]) if len(H1)>0 else 0.0
    local_scores[i] = pers_H1

    # PCA
    pca = PCA(n_components=min(5, X.shape[1]))
    pca.fit(neigh)
    svals = pca.singular_values_
    dim_ratio[i] = (svals[1]/svals[0]) if len(svals)>1 and svals[0]>1e-12 else 0.0

# Heuristic thresholds:
thr_pers = np.percentile(local_scores, 20)  # bottom 20%: weaker H1 locally
thr_dimr = np.percentile(dim_ratio, 20)     # bottom 20%: more 1D-like
singular_flags = (local_scores < thr_pers) | (dim_ratio < thr_dimr)
print("Singular-locus candidates:", np.sum(singular_flags))
```

**Visualize candidates** (via 3D embedding in the next section); expected to trace out a **closed 1D curve** (the rational normal quartic inside \(\mathbb{RP}^4\)).

---

## 4) 3D embeddings & **mesh** generation

To produce meshes, first **embed to 3D** (UMAP preserves neighborhoods well), then build an **alpha shape** or **ball‑pivoting** surface.

```python
import umap
emb = umap.UMAP(n_components=3, metric='euclidean', n_neighbors=50, min_dist=0.05, random_state=42).fit_transform(X)

# Save a scatter and color by singularity score
import plotly.express as px
fig = px.scatter_3d(x=emb[:,0], y=emb[:,1], z=emb[:,2],
                    color=singular_flags.astype(int), opacity=0.7, size_max=2)
fig.write_html("embedding3d.html")
```

**Mesh via Open3D (Ball Pivoting)**

```python
import open3d as o3d

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(emb.astype(np.float64))

# Estimate normals (needed for ball pivoting)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))

radii = [0.02, 0.04, 0.08]  # tune based on scale of emb
rec = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    pcd, o3d.utility.DoubleVector(radii)
)
o3d.io.write_triangle_mesh("surface_mesh_bpa.ply", rec)
```

**Alternative mesh**: alpha shape in 3D (via GUDHI AlphaComplex on the 3D embedding).

```python
import gudhi as gd

ac3 = gd.AlphaComplex(points=emb.astype(np.float64))
st3 = ac3.create_simplex_tree()
st3.compute_persistence()
# Extract the 2-skeleton triangles around chosen alpha radius
# (You can filter simplices by filtration value to get a watertight mesh at a chosen scale)
```

**Visualizing the singular locus**: export candidate points as a PLY with a distinct color.

```python
cand = emb[singular_flags]
import open3d as o3d
pcd_cand = o3d.geometry.PointCloud()
pcd_cand.points = o3d.utility.Vector3dVector(cand.astype(np.float64))
o3d.io.write_point_cloud("singular_curve_candidates.ply", pcd_cand)
```

---

## 5) Sanity checks against expected homology

- **H\_0**: one long bar (connected).  
- **H\_1**: two persistent classes suggest \(b_1\approx 2\) (torus‑like real locus).  
- **H\_2**: possibly one moderately persistent 2‑cycle in good samplings.  
- If your sampling heavily includes points near the singular curve, you may see **extra short‑lived cycles** and noisy features—filter them by persistence thresholding.

**Parameter tuning tips**
- Use **antipodal distances** to respect projective identification.  
- For VR/Rips: adjust maximum scale (e.g. 95‑percentile of pairwise distances).  
- Subsample to ~2000 points for Alpha complex in high dimension.  
- Local analysis: \(k=30\)–\(100\), try multiple radii/ks and aggregate flags by majority vote.

---

## 6) Reproducible runner

Create `run_tda.py` from the snippets above and add CLI args:
```bash
python run_tda.py --csv sampled_points.csv --complex False --neighbors 50 --outdir results/
```

Outputs:
- `diagrams.png` (VR persistence)  
- `embedding3d.html` (interactive)  
- `surface_mesh_bpa.ply` (mesh)  
- `singular_curve_candidates.ply` (suspected Veronese quartic)  

---

## 7) Notes & caveats

- This TDA recovers **real**-topology features from real samples. Complex Betti numbers are not directly visible in a real point cloud.  
- The singular locus is **1D**; our detectors (local homology + PCA) mark a **1D ribbon** of candidates. A spline fit through candidates can recover a smooth proxy for the rational normal quartic.  
- Mapper (optional) can summarize global structure; choose filter \=(e.g.) first two PCs and a density lens.

---

## References

- Edelsbrunner, Harer — *Computational Topology* (AMS, 2010).  
- Otter, Porter, Tillmann, Ulmer, Zomorodian — *A roadmap for the computation of persistent homology*, EPJ Data Science (2017).  
- Bendich, Wang, Mukherjee — *Inference of Stratification in Noisy Point Cloud Data*, Information and Inference (2012).  
- Edelsbrunner, Mücke — *Three-dimensional alpha shapes*, TOG (1994).  
- Singh, Mémoli, Carlsson — *Mapper*, Eurographics (2007).
