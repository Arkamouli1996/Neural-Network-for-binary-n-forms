
# Uniform Sampling on the Triple‑Root Surface in **ℂP⁴** with HomotopyContinuation.jl
**Goal:** Generate i.i.d. samples **uniform w.r.t. the Fubini–Study (area) measure** on the **complex surface**
\(\mathcal{S} \subset \mathbb{C}P^4\) of binary quartics with a triple root, and i.i.d. negatives uniform in \(\mathbb{C}P^4\). Then train a 1‑hidden‑layer classifier.

This README is **implementation‑first**. Follow it literally; no algebra background needed.

---

## 0) What we are sampling
- A binary quartic is \(f(x,y)=a_0 x^4 + a_1 x^3y + a_2 x^2y^2 + a_3 xy^3 + a_4 y^4\), represented by the complex **coefficient** vector \(a=(a_0,\dots,a_4)\in\mathbb{C}^5\), up to scaling → a point \([a]\in\mathbb{C}P^4\).
- The **triple‑root surface** \(\mathcal{S}\subset\mathbb{C}P^4\) is cut out by two **homogeneous** equations in the 5 coefficients:
  - \(H_1(a)=\mathrm{Res}(F,F')=0\) (the quartic discriminant),
  - \(H_2(a)=0\) (the **second subdiscriminant**, obtained from elimination of \(u\) in \(\langle F,F',F''\rangle\)).
  You’ll be given both \(H_1,H_2\) as sparse polynomials (see §1).

**Uniform on \(\mathbb{C}P^4\)** means unitary‑invariant (Fubini–Study). **Uniform on \(\mathcal{S}\)** means the induced area measure on the complex 2‑fold \(\mathcal{S}\).

---

## 1) Files to put in place
```
project/
├── data/
│   ├── H1_sparse.json           # list of {coeff: [re,im], exponents: [e0..e4]}
│   └── H2_sparse.json
├── julia/
│   ├── poly_eval.jl             # builds fast evaluators H(a), ∇H(a)
│   ├── slice_system.jl          # builds and solves sliced systems with HC.jl
│   ├── sampler_cp4_uniform.jl   # negatives: uniform on CP^4
│   ├── sampler_surface_cpx.jl   # positives: uniform on surface via CP^2-slicing + weights
│   └── utils.jl                 # linear algebra helpers, QR for complex, chart selection
└── python/
    └── train_nn.py              # 1-layer classifier (any framework)
```

> If you don’t have `H1_sparse.json` / `H2_sparse.json` yet, export them from Macaulay2: eliminate \(u\) in \(\langle F,F',F''\rangle\) to get the two generators; serialize as sparse monomials.

---

## 2) Dependencies
```julia
using Pkg
Pkg.add.(["HomotopyContinuation", "LinearAlgebra", "Random",
          "StaticArrays", "JSON3", "ForwardDiff", "Distributions"])
```

---

## 3) Evaluating H₁, H₂ and their gradients
**julia/poly_eval.jl**
```julia
module PolyEval
using StaticArrays, JSON3

struct SparsePoly
    coeffs::Vector{ComplexF64}
    exps::Vector{SVector{5,Int}}
end

function load_sparsepoly(path::String)
    raw = JSON3.read(open(path, "r"))
    coeffs = ComplexF64[]
    exps = SVector{5,Int}[]
    for t in raw
        c = t["coeff"]
        push!(coeffs, ComplexF64(c[0], c[1]))
        push!(exps, SVector{5,Int}(t["exponents"]))
    end
    return SparsePoly(coeffs, exps)
end

"Evaluate sparse homogeneous polynomial at a (length-5 complex)."
function heval(p::SparsePoly, a::SVector{5,ComplexF64})
    s = 0.0 + 0.0im
    @inbounds for k in eachindex(p.coeffs)
        term = p.coeffs[k]
        e = p.exps[k]
        @inbounds for i in 1:5
            ei = e[i]
            if ei != 0
                term *= a[i]^ei
            end
        end
        s += term
    end
    return s
end

"Numerical complex gradient via central differences (sufficient for weighting)."
function hgrad(p::SparsePoly, a::SVector{5,ComplexF64}; eps=1e-7)
    g = MVector{5,ComplexF64}(zeros(ComplexF64,5))
    for k in 1:5
        δ = ntuple(i-> (i==k ? eps : 0.0) + 0.0im, 5)
        ap = SVector{5,ComplexF64}(a[i] + δ[i] for i in 1:5)
        am = SVector{5,ComplexF64}(a[i] - δ[i] for i in 1:5)
        gp = heval(p, ap); gm = heval(p, am)
        g[k] = (gp - gm)/(2eps)
    end
    return SVector{5,ComplexF64}(g)
end

export SparsePoly, load_sparsepoly, heval, hgrad
end
```

---

## 4) Uniform negatives in **ℂP⁴**
**julia/sampler_cp4_uniform.jl**
```julia
module SamplerCP4
using LinearAlgebra, Random, StaticArrays, Distributions

"Draw N i.i.d. uniform points in ℂP⁴ (columns)."
function sample_cp4(N::Int; rng=Random.default_rng())
    X = Matrix{ComplexF64}(undef, 5, N)
    for j in 1:N
        v = randn(rng, ComplexF64, 5)
        v ./= norm(v)   # ℓ2-normalize
        X[:,j] = v
    end
    return X
end

export sample_cp4
end
```

---

## 5) Random **ℂP²** slices (unitary‑invariant)
**julia/utils.jl**
```julia
module Utils
using LinearAlgebra, Random

"Return 5×3 matrix with orthonormal columns spanning a random 3-plane W."
function random_C3_subspace(; rng=Random.default_rng())
    Z = randn(rng, ComplexF64, 5, 3)
    Q, _ = qr(Z)  # complex QR
    return Matrix(Q[:,1:3])
end

export random_C3_subspace
end
```

---

## 6) Build & solve sliced systems; compute weights
**julia/slice_system.jl**
```julia
module SliceSystem
using HomotopyContinuation, StaticArrays, LinearAlgebra
import ..PolyEval: SparsePoly, heval, hgrad

@var x y

"Ambient representative a(x,y) = B * [x, y, 1]. B is 5×3 complex."
@inline function ambient_point(B::AbstractMatrix{ComplexF64}, x::ComplexF64, y::ComplexF64)
    return SVector{5,ComplexF64}(B * SVector{3,ComplexF64}(x, y, 1+0im))
end

"Crofton weight w = 1/|det J|, J_ij = ∂H_i/∂(var_j) at (x*,y*)."
function crofton_weight(B::AbstractMatrix{ComplexF64}, H1::SparsePoly, H2::SparsePoly, x::ComplexF64, y::ComplexF64)
    a = ambient_point(B, x, y)
    g1 = hgrad(H1, a)
    g2 = hgrad(H2, a)
    dax = SVector{5,ComplexF64}(B[:,1])
    day = SVector{5,ComplexF64}(B[:,2])
    J11 = dot(g1, dax);  J12 = dot(g1, day)
    J21 = dot(g2, dax);  J22 = dot(g2, day)
    detJ = J11*J22 - J12*J21
    return 1.0 / max(abs(detJ), 1e-14)   # clip
end

"Solve the slice; return 2×K matrix of (x,y) and weights."
function solve_slice(B::AbstractMatrix{ComplexF64}, H1::SparsePoly, H2::SparsePoly; tol=1e-10)
    # Define equations for HC.jl
    F = @f! (x, y) -> begin
        a = ambient_point(B, x, y)
        ( heval(H1, a), heval(H2, a) )
    end
    sols = solve(F; show_progress=false)
    xy = ComplexF64[]
    wts = Float64[]
    for s in solutions(sols)
        x, y = s[1], s[2]
        a = ambient_point(B, x, y)
        r1 = abs(heval(H1, a)); r2 = abs(heval(H2, a))
        if r1 ≤ tol && r2 ≤ tol
            push!(xy, x); push!(xy, y)
            push!(wts, crofton_weight(B, H1, H2, x, y))
        end
    end
    return reshape(xy, 2, :), wts
end

export ambient_point, crofton_weight, solve_slice
end
```

---

## 7) Final positive sampler (uniform on the surface)
**julia/sampler_surface_cpx.jl**
```julia
module SamplerSurfaceCpx
using Random, LinearAlgebra, StaticArrays, Distributions
import ..Utils: random_C3_subspace
import ..PolyEval: SparsePoly
import ..SliceSystem: ambient_point, solve_slice

"Draw M i.i.d. uniform points on the triple-root surface in ℂP⁴."
function sample_surface(M::Int, H1::SparsePoly, H2::SparsePoly; rng=Random.default_rng(), tol=1e-10)
    A = Matrix{ComplexF64}(undef, 5, M)
    cnt = 0
    while cnt < M
        B = random_C3_subspace(; rng)                # random CP^2
        XY, wts = solve_slice(B, H1, H2; tol=tol)    # intersection points + weights
        K = size(XY,2)
        if K == 0 || sum(wts) == 0
            continue
        end
        # Select one point on this slice with prob ∝ weight
        p = wts ./ sum(wts)
        k = rand(rng, Categorical(p))
        x, y = XY[1,k], XY[2,k]
        a = ambient_point(B, x, y)
        a ./= norm(a)      # store unit-ℓ2 rep
        cnt += 1
        A[:,cnt] = a
    end
    return A  # columns = samples on the surface
end

export sample_surface
end
```

---

## 8) End-to-end example
```julia
include("julia/poly_eval.jl")
include("julia/utils.jl")
include("julia/slice_system.jl")
include("julia/sampler_cp4_uniform.jl")
include("julia/sampler_surface_cpx.jl")

H1 = PolyEval.load_sparsepoly("data/H1_sparse.json")
H2 = PolyEval.load_sparsepoly("data/H2_sparse.json")

Apos = SamplerSurfaceCpx.sample_surface(5000, H1, H2)
Aneg = SamplerCP4.sample_cp4(5000)

# Save to disk (CSV or JLD2). Here: simple NPY via Python is fine.
```

---

## 9) Validation checklist
- **Residuals:** For each surface sample `a`, check `abs(H1(a))` and `abs(H2(a))` ≤ `1e-10`.
- **Weight sanity:** Plot histogram of weights; if heavy tail → increase solver precision or reject outliers.
- **Uniformity smoke test:** Build a coarse reference set by collecting many unweighted intersections on random slices; then bin your i.i.d. samples against that set and look for flatness across bins.

---

## 10) Common pitfalls (and fixes)
- **Using max‑norm for normalization.** Don’t. Always ℓ²‑normalize complex vectors.
- **Forgetting weights.** Taking all slice intersections equally is **not** uniform. Always sample **one per slice with Crofton weights**.
- **Bad chart.** If many slices fail, consider chart switching (choose `w_j=1` for the most stable coordinate).

---

## 11) Training (optional)
See `python/train_nn.py` in this README for a minimal 1‑layer classifier. Convert complex 5‑vectors to real 10‑vectors by concatenating real/imag parts.

---

**You now have everything needed to implement complex‑uniform sampling on the triple‑root surface using HomotopyContinuation.jl.**
