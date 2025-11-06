# Uniform Sampling on the Triple‑Root Surface in \(\mathbb{P}^4\) with HomotopyContinuation.jl

This README is a **coding playbook** for a Julia agent to **sample points uniformly** (in the sense of *kinematic measure* induced by random linear slices) on the surface
\[
S \;=\;\{[a_0:\dots:a_4]\in \mathbb{P}^4 \mid E_2(a)=0,\; E_3(a)=0\},
\]
where \(E_2\) is a quadric and \(E_3\) is a cubic whose common zero set is the **triple‑root locus of binary quartics** (quartics of the form \(L^3M\)). The scheme‑theoretic degree is \(\deg S = 6\). The surface is singular along the rational normal quartic \(\nu_4(\mathbb{P}^1)=\{L^4\}\subset \mathbb{P}^4\).

The strategy follows **numerical algebraic geometry (NAG)**: sample random linear sections of complementary codimension (two independent linear equations in \(\mathbb{P}^4\), i.e. a random \(\mathbb{P}^2\)), compute the intersection \(S\cap \mathbb{P}^2\) (six points over \(\mathbb{C}\), counting multiplicity), and take these intersection points as **i.i.d. samples from the kinematic distribution** on \(S\). See the references for why this is the right notion of uniformity for algebraic manifolds (and how to adapt on singular sets).

> TL;DR for implementers: **each sample = new random slice**. Avoid reusing the same slice and avoid monodromy shuffles if you want independence.


---

## 0) Install & set up

```julia
using Pkg
Pkg.add(["HomotopyContinuation", "Random", "LinearAlgebra", "Statistics"])

using HomotopyContinuation, Random, LinearAlgebra, Statistics
```

You will need the two defining equations. If you have them in a text file, paste them below. We work in homogeneous coordinates \([a_0:\dots:a_4]\). For computation, we impose a random **projective patch** \(u\cdot a=1\) to avoid bias toward any fixed affine chart.


---

## 1) Define the surface system \((E_2,E_3)\)

Edit the two expressions `E2_expr` and `E3_expr` to match your generators. (They are **polynomials in** `a0,a1,a2,a3,a4`.)

```julia
# === VARIABLES ===
@polyvar a0 a1 a2 a3 a4

# === DEFINE YOUR EQUATIONS ===
# Replace the right-hand sides with your actual polynomials:
E2_expr = a0*a4 - 4*a1*a3 + 3*a2^2   # <-- placeholder quadric
E3_expr = a0*a2*a4 + 2*a1*a2*a3 - a2^3 - a0*a3^2 - a1^2*a4  # <-- placeholder cubic

function surface_system()
    System([E2_expr, E3_expr])
end
```

> If you’d rather load from a file, parse the line(s) into expressions and `Meta.parse` them into `Polynomial`s, or just paste the explicit polynomials here for reproducibility.


---

## 2) Random projective slices that realize kinematic uniformity

A **uniform random sample** (over \(\mathbb{C}\)) is obtained by intersecting \(S\) with a **random \(\mathbb{P}^2\subset\mathbb{P}^4\)**. In coordinates, pick a random **unit vector** \(u\in\mathbb{C}^5\) (patch constraint \(u\cdot a = 1\)), then pick **two independent random linear forms** \(\ell_1,\ell_2\) (rows of a random complex Gaussian matrix, orthonormalized for stability). The slice is
\([u\cdot a - 1 = 0,\ \ell_1(a)=0,\ \ell_2(a)=0].\)

> This randomization corresponds to drawing from the **Haar‑invariant** measure on the complex Grassmannian of 2‑planes inside \(\mathbb{P}^4\).

```julia
function rand_unitvec_complex(n)
    v = randn(ComplexF64, n) .+ im*randn(ComplexF64, n)
    v / norm(v)
end

function random_slice_constraints()
    # random patch: u ⋅ a = 1
    u = rand_unitvec_complex(5)
    patch = u[1]*a0 + u[2]*a1 + u[3]*a2 + u[4]*a3 + u[5]*a4 - 1

    # two random independent linear forms defining a P^2 in the patch
    A = randn(ComplexF64, 2, 5) .+ im*randn(ComplexF64, 2, 5)
    Q, _ = qr(A')  # orthonormal columns
    l1 = Q[:,1]'*[a0,a1,a2,a3,a4]
    l2 = Q[:,2]'*[a0,a1,a2,a3,a4]

    return (patch, l1, l2)
end
```

**Why patch+two linear forms?** In affine coordinates the surface has dimension 2, so intersecting with two independent hyperplanes yields a **0‑dimensional** set of \(\deg S=6\) points (over \(\mathbb{C}\), counted with multiplicities). The random patch avoids biasing toward \(a_0\neq 0\) and eliminates “solutions at infinity.”


---

## 3) Solving a slice and collecting a sample

We assemble the polynomial system `[E2, E3, patch, l1, l2]` and call `solve`. We return the **projective points** (normalize each solution vector).

```julia
function sample_once_complex(; endgame=:cauchy, seed=nothing)
    F = surface_system()
    patch, l1, l2 = random_slice_constraints()
    sys = System([F.polys... , patch, l1, l2])

    opts = solve(;
        start_system = :polyhedral,  # robust default; can also try :total_degree
        show_progress = false,
        max_steps = 10_000,
        # endgame to stabilize near singular intersections:
        endgame = (endgame == :cauchy ? CauchyEndgame() : nothing),
        # reproducibility
        seed = seed,
    )
    R = solve(sys; opts...)

    sols = solutions(R)  # Complex solutions
    # Convert to homogeneous projective points in C^5 by solving the linear system (variables are a0..a4).
    # In HomotopyContinuation, solutions return dictionaries when using System; obtain in variable order:
    vars = variables(sys)
    ord = Dict(v=>i for (i,v) in enumerate([a0,a1,a2,a3,a4]))
    pts = [begin
        x = zeros(ComplexF64,5)
        for (v,val) in zip(vars, s)
            if haskey(ord, v); x[ord[v]] = val; end
        end
        # undo patch: the 'patch' equation fixed u⋅a = 1, so x is already affine. Normalize to projective:
        x ./ maximum(abs.(x))  # scale invariant representative
    end for s in sols]

    return pts
end

function sample_k_complex(k::Integer)
    vcat([sample_once_complex() for _ in 1:k]...)
end
```

**Real samples**: Random **real** slices (draw real Gaussians) + filter real solutions.

```julia
function sample_once_real(; atol=1e-8)
    F = surface_system()
    # real patch/slice
    u = (randn(5)); u = u/norm(u)
    patch = u[1]*a0 + u[2]*a1 + u[3]*a2 + u[4]*a3 + u[5]*a4 - 1
    A = randn(2,5)
    Q,_ = qr(A')
    l1 = Q[:,1]'*[a0,a1,a2,a3,a4]
    l2 = Q[:,2]'*[a0,a1,a2,a3,a4]

    sys = System([F.polys... , patch, l1, l2])
    R = solve(sys; endgame=CauchyEndgame(), show_progress=false)
    solsR = real_solutions(R; atol=atol)

    vars = variables(sys)
    ord = Dict(v=>i for (i,v) in enumerate([a0,a1,a2,a3,a4]))
    pts = [begin
        x = zeros(Float64,5)
        for (v,val) in zip(vars, s)
            if haskey(ord, v); x[ord[v]] = real(val); end
        end
        x ./ maximum(abs.(x))
    end for s in solsR]

    return pts
end
```

> Over \(\mathbb{R}\), the number of real points in a slice **varies** (anywhere from 0 to 6). For *uniform real sampling*, keep drawing random real slices until you collect enough points; this realizes the **real kinematic measure** on \(S(\mathbb{R})\).


---

## 4) Sampling **near the singular locus** \(\nu_4(\mathbb{P}^1)\)

The singular curve consists of the points \([L^4]\). We can target neighborhoods of this curve in two complementary ways:

### (A) **Conditioned slices** (projective, unbiased in the kinematic sense)

1. Pick a random point \(p\) on the singular curve by drawing \(L=[u:v]\in\mathbb{P}^1\) and setting the binary quartic coefficients of \(L^4\).
2. Choose a random slice \(\mathbb{P}^2\) that **passes ε‑close** to \(p\) (or contains \(p\) and then randomly rotate within the stabilizer).  
3. Solve the sliced system with a **Cauchy endgame** and **deflation** if needed.

```julia
# coefficients of (ux+vy)^4 in the monomial basis {x^4, x^3 y, x^2 y^2, x y^3, y^4}
binom4 = [1,4,6,4,1]
function veronese4(u,v)
    ComplexF64.([binom4[1]*u^4, binom4[2]*u^3*v, binom4[3]*u^2*v^2, binom4[4]*u*v^3, binom4[5]*v^4])
end

function sample_near_singular(; eps=1e-3)
    # step 1: pick random L
    U = randn() + im*randn(); V = randn() + im*randn()
    p = veronese4(U,V); p /= maximum(abs.(p))

    # step 2: random slice through/near p: enforce two real/complex linear constraints close to p
    # construct two random directions orthonormal to p, then offset by eps
    B = nullspace([p'] )  # 5×4 basis with columns spanning orthogonal complement (numerical)
    w1,w2 = B[:,1], B[:,2]
    # plane: {a | (w1⋅a)=eps1, (w2⋅a)=eps2} with small eps's
    eps1 = eps*(randn()+im*randn()); eps2 = eps*(randn()+im*randn())
    l1 = w1[1]*a0 + w1[2]*a1 + w1[3]*a2 + w1[4]*a3 + w1[5]*a4 - eps1
    l2 = w2[1]*a0 + w2[2]*a1 + w2[3]*a2 + w2[4]*a3 + w2[5]*a4 - eps2

    # random patch for projective normalization
    u = rand_unitvec_complex(5)
    patch = u[1]*a0 + u[2]*a1 + u[3]*a2 + u[4]*a3 + u[5]*a4 - 1

    sys = System([surface_system().polys... , patch, l1, l2])
    R = solve(sys; endgame=CauchyEndgame(), show_progress=false)
    return solutions(R)
end
```

This maintains the **slice‑based notion of uniformity** while biasing location *conditionally* to the neighborhood of the singular curve.

### (B) **Parametric biasing** (efficient, but must reweight to be globally uniform)

Use the parametrization \( (L,M)\in \mathbb{P}^1\times\mathbb{P}^1 \mapsto L^3M \) to generate samples with \(M\) close to \(L\) (e.g. set \(M=L+\delta\) with \(|\delta|\ll 1\)), then **push forward** to coefficients. This is computationally cheap and excellent for probing numerics near singularities, but it **does not** sample uniformly on \(S\) unless you **reweight by the Jacobian density** of the map. Use it for stress‑tests and then fall back to (A) for unbiased sampling.


---

## 5) Practical numerics: endgames, deflation, precision

- Always enable a robust **endgame** (e.g., `CauchyEndgame()`) when slicing near the singular curve; this stabilizes limits of paths terminating at multiplicity‑>1 points.
- If you need to **resolve multiplicity** or track branches through embedded components, apply **deflation** (isosingular deflation). HomotopyContinuation.jl exposes singularity diagnostics (`nsingular`, `singular`) and you can incorporate deflation homotopies when needed.
- If conditioning is poor, **raise precision**:
  ```julia
  setprecision(BigFloat, 256)
  ```
  and rebuild the system with `BigFloat` coefficients.

- For **independent samples**, regenerate a **fresh random slice** every time. Monodromy loops are great for exploring a fiber but introduce dependence.


---

## 6) Real vs. complex sampling — what changes?

- Over **\(\mathbb{C}\)**: every random slice meets \(S\) in exactly \(\deg S=6\) points (counted with multiplicities) almost surely. Drawing each slice Haar‑random on the Grassmannian gives **i.i.d. samples** from the complex kinematic measure on \(S\).
- Over **\(\mathbb{R}\)**: a random real slice meets \(S(\mathbb{R})\) in a **random number** of points (from 0 to 6). To obtain samples distributed according to the **real kinematic measure**, repeatedly draw **independent random real slices** and **collect the real intersection points**. (Discarding complex conjugate pairs introduces **no bias** for the real kinematic measure.)
- Near singularities, **endgames + deflation** are more frequently required in the real case (paths may approach singular real points tangentially).


---

## 7) Minimal end‑to‑end example

```julia
# 1) define the surface system
F = surface_system()

# 2) draw 100 complex samples (≈ i.i.d. kinematic)
samples = [sample_once_complex() for _ in 1:100]
# flatten:
Samps = reduce(vcat, samples)  # each row is a length-5 ComplexF64 vector up to projective scale

# 3) draw real samples until you have ~100 points on S(R)
reals = []
while length(reals) < 100
    pts = sample_once_real()
    append!(reals, pts)
end
```

**Quality checks**
- Verify degree by counting solutions per slice (should be 6 a.s.).
- `nsingular` should be zero for generic slices; nonzero indicates near‑singular slicing—expected if you conditioned on the singular curve.
- Use a second set of random slices to **Chi‑square** test invariance under unitary rotations (simple invariant features like distribution of \(|a_i|^2\) should match across independently drawn slices).


---

## 8) Notes on “uniformity” and measure

The slice‑based method induces the **kinematic (Crofton) measure** on \(S\): pick a random \(\mathbb{P}^2\) (Haar on \(G(3,5)\)), take intersection points. For **smooth** \(S\), this coincides with the distribution obtained by pushing forward the Fubini–Study surface area density under generic local charts. On **singular** \(S\), kinematic sampling is still well‑defined and naturally discounts multiplicity unless you explicitly count multiplicities.


---

## References (theory & software)

- **Witness sets & sampling**:  
  *Bates–Hauenstein–Sommese–Wampler*, _Numerically Solving Polynomial Systems with Bertini_, SIAM (2013), esp. witness sets, kinematic sampling, endgames.  
  *Sottile*, “General witness sets for numerical algebraic geometry”, arXiv:2002.00180 (2020).  
  *Hauenstein–Oeding–others*, “Multiprojective witness sets and a trace test”, arXiv:1507.07069 (2015).

- **Random points via random slices**:  
  *Breiding–Marigliano*, “Random points on an algebraic manifold”, SIAM J. Appl. Alg. Geom. (2019). Preprint: arXiv:1810.06271.  
  Background on integral geometry and Crofton/kinematic formulas: *Santaló*, _Integral Geometry and Geometric Probability_, Cambridge (2nd ed., 2011).

- **Singularities, endgames, deflation**:  
  *Hauenstein–Wampler*, “Isosingular sets and deflation”, Found. Comput. Math. (2013).  
  *Bates–Beltrán–Hauenstein–Sommese*, “A parallel endgame”, (2010).  
  *Akoglu–Beltrán–Hauenstein–Telen*, “Certifying solutions to overdetermined and singular polynomial systems”, (2018).

- **Real algebraic sampling**:  
  *Brake–Bates–Hauenstein–Sommese–Wampler*, “Software for one‑ and two‑dimensional real algebraic sets (Bertini_real)”, and “On computing a cell decomposition of a real surface containing singular curves”.

- **HomotopyContinuation.jl**:  
  Breiding–Timme, “HomotopyContinuation.jl: A Package for Homotopy Continuation in Julia” (ICMS 2018).  
  Official docs: witness sets, endgames, real solutions.

Happy sampling.
