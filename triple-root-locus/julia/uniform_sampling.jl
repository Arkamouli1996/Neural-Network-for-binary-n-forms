# Uniform Sampling on the Triple-Root Surface
# Using HomotopyContinuation.jl for numerical algebraic geometry

using HomotopyContinuation
using Random
using LinearAlgebra
using Statistics

println("Loading HomotopyContinuation.jl and dependencies...")

# === VARIABLES ===
@polyvar a0 a1 a2 a3 a4

# === DEFINE THE SURFACE EQUATIONS ===
# From triple_root_generators.txt:
# E1 = a2^2 - 3*a1*a3 + 12*a0*a4
# E2 = a1*a2*a3 - 9*a0*a3^2 - 9*a1^2*a4 + 32*a0*a2*a4

E1_expr = a2^2 - 3*a1*a3 + 12*a0*a4
E2_expr = a1*a2*a3 - 9*a0*a3^2 - 9*a1^2*a4 + 32*a0*a2*a4

println("Surface equations loaded:")
println("E1 = ", E1_expr)
println("E2 = ", E2_expr)

function surface_system()
    System([E1_expr, E2_expr])
end

# === RANDOM PROJECTIVE SLICES ===
function rand_unitvec_complex(n)
    v = randn(ComplexF64, n) .+ im*randn(ComplexF64, n)
    v / norm(v)
end

function random_slice_constraints()
    # Random patch: u ⋅ a = 1
    u = rand_unitvec_complex(5)
    patch = u[1]*a0 + u[2]*a1 + u[3]*a2 + u[4]*a3 + u[5]*a4 - 1

    # Two random independent linear forms defining a P^2 in the patch
    A = randn(ComplexF64, 2, 5) .+ im*randn(ComplexF64, 2, 5)
    Q, _ = qr(A')  # Orthonormal columns
    l1 = Q[:,1]' * [a0, a1, a2, a3, a4]
    l2 = Q[:,2]' * [a0, a1, a2, a3, a4]

    return (patch, l1, l2)
end

# === COMPLEX SAMPLING ===
function sample_once_complex(; seed=nothing, verbose=false)
    F = surface_system()
    patch, l1, l2 = random_slice_constraints()
    sys = System([expressions(F)..., patch, l1, l2])

    if verbose
        println("Solving system with $(length(expressions(sys))) equations...")
    end

    # Set up solver options
    R = solve(sys; 
        start_system = :polyhedral,
        show_progress = verbose
    )

    if verbose
        println("Found $(nresults(R)) results, $(nsolutions(R)) nonsingular solutions")
    end

    sols = solutions(R)
    
    # Convert to homogeneous projective points
    # Solutions are already in [a0, a1, a2, a3, a4] order
    pts = [begin
        x = ComplexF64.(s)
        # Normalize to projective representative
        max_abs = maximum(abs.(x))
        if max_abs > 1e-10
            x ./ max_abs
        else
            x
        end
    end for s in sols]

    return pts, R
end

function sample_k_complex(k::Integer; verbose=false)
    all_pts = Vector{Vector{ComplexF64}}()
    
    for i in 1:k
        if verbose && i % 10 == 0
            println("Sampling slice $i/$k...")
        end
        pts, _ = sample_once_complex(verbose=false)
        append!(all_pts, pts)
    end
    
    if verbose
        println("Collected $(length(all_pts)) total points from $k slices")
    end
    
    return all_pts
end

# === REAL SAMPLING ===
function sample_once_real(; atol=1e-8, verbose=false)
    F = surface_system()
    
    # Real patch/slice
    u = randn(5)
    u = u / norm(u)
    patch = u[1]*a0 + u[2]*a1 + u[3]*a2 + u[4]*a3 + u[5]*a4 - 1
    
    A = randn(2, 5)
    Q, _ = qr(A')
    l1 = Q[:,1]' * [a0, a1, a2, a3, a4]
    l2 = Q[:,2]' * [a0, a1, a2, a3, a4]

    sys = System([expressions(F)..., patch, l1, l2])
    R = solve(sys; show_progress=verbose)
    
    solsR = real_solutions(R; tol=atol)

    if verbose
        println("Found $(length(solsR)) real solutions out of $(nsolutions(R)) total")
    end

    # Convert to projective points
    pts = [begin
        x = Float64.(s)
        max_abs = maximum(abs.(x))
        if max_abs > 1e-10
            x ./ max_abs
        else
            x
        end
    end for s in solsR]

    return pts, R
end

function sample_k_real_points(k::Integer; verbose=false)
    reals = Vector{Vector{Float64}}()
    slice_count = 0
    
    while length(reals) < k
        slice_count += 1
        if verbose && slice_count % 10 == 0
            println("Slice $slice_count: collected $(length(reals))/$k real points so far...")
        end
        pts, _ = sample_once_real(verbose=false)
        append!(reals, pts)
    end
    
    if verbose
        println("Collected $(length(reals)) real points from $slice_count slices")
    end
    
    return reals[1:k], slice_count
end

# === SAMPLING NEAR SINGULAR LOCUS ===
# Veronese embedding: [u:v] → [u^4 : u^3v : u^2v^2 : uv^3 : v^4]
binom4 = [1, 4, 6, 4, 1]

function veronese4(u, v)
    ComplexF64.([
        binom4[1]*u^4,
        binom4[2]*u^3*v,
        binom4[3]*u^2*v^2,
        binom4[4]*u*v^3,
        binom4[5]*v^4
    ])
end

function sample_near_singular(; eps=1e-3, verbose=false)
    # Step 1: Pick random L = [U:V]
    U = randn() + im*randn()
    V = randn() + im*randn()
    p = veronese4(U, V)
    p /= maximum(abs.(p))

    if verbose
        println("Target singular point: ", p)
    end

    # Step 2: Random slice through/near p
    # Construct two random directions orthonormal to p
    # Create matrix with p as a row
    p_matrix = reshape(p, 1, 5)
    B = nullspace(p_matrix)
    w1, w2 = B[:,1], B[:,2]
    
    # Plane: {a | (w1⋅a)=eps1, (w2⋅a)=eps2}
    eps1 = eps * (randn() + im*randn())
    eps2 = eps * (randn() + im*randn())
    l1 = w1[1]*a0 + w1[2]*a1 + w1[3]*a2 + w1[4]*a3 + w1[5]*a4 - eps1
    l2 = w2[1]*a0 + w2[2]*a1 + w2[3]*a2 + w2[4]*a3 + w2[5]*a4 - eps2

    # Random patch for projective normalization
    u = rand_unitvec_complex(5)
    patch = u[1]*a0 + u[2]*a1 + u[3]*a2 + u[4]*a3 + u[5]*a4 - 1

    sys = System([expressions(surface_system())..., patch, l1, l2])
    R = solve(sys; show_progress=verbose)
    
    sols = solutions(R)
    pts = [ComplexF64.(s) ./ maximum(abs.(ComplexF64.(s))) for s in sols]
    
    return pts, R
end

# === MAIN DEMO ===
function demo_sampling()
    println("\n" * "="^60)
    println("DEMO: Uniform Sampling on Triple-Root Surface")
    println("="^60)
    
    # Test 1: Single complex slice
    println("\n1. Testing single complex slice (should give ~6 points):")
    pts, R = sample_once_complex(verbose=true)
    println("   → Got $(length(pts)) points")
    println("   → First point: ", pts[1])
    
    # Test 2: Multiple complex slices
    println("\n2. Sampling 5 complex slices:")
    all_pts = sample_k_complex(5, verbose=true)
    println("   → Total points: $(length(all_pts))")
    
    # Test 3: Real sampling
    println("\n3. Testing real sampling (1 slice):")
    real_pts, R_real = sample_once_real(verbose=true)
    println("   → Got $(length(real_pts)) real points")
    if length(real_pts) > 0
        println("   → First real point: ", real_pts[1])
    end
    
    # Test 4: Collect multiple real points
    println("\n4. Collecting 20 real points:")
    real_collection, n_slices = sample_k_real_points(20, verbose=true)
    println("   → Required $n_slices slices to collect 20 real points")
    
    # Test 5: Near singular locus
    println("\n5. Sampling near singular locus:")
    sing_pts, R_sing = sample_near_singular(eps=1e-3, verbose=true)
    println("   → Got $(length(sing_pts)) points near singular locus")
    
    println("\n" * "="^60)
    println("All tests completed successfully!")
    println("="^60)
end

# Run the demo
if abspath(PROGRAM_FILE) == @__FILE__
    demo_sampling()
end
