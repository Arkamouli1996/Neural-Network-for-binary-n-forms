# Generate dataset for neural network training using UNIFORM SAMPLING
# This implementation follows the README's prescribed method with:
# - Crofton weights for uniform measure on the complex surface
# - Random CP^2 slices (unitary-invariant)
# - One point per slice sampled with probability ∝ weight
#
# Usage: julia generate_data.jl [n_positive] [n_negative]
# Example: julia generate_data.jl 2000 2000

# Load all modules following the README architecture
include("poly_eval.jl")
include("utils.jl")
include("slice_system.jl")
include("sampler_cp4_uniform.jl")
include("sampler_surface_cpx.jl")

using DelimitedFiles
using Random
using LinearAlgebra
using .PolyEval
using .Utils
using .SliceSystem
using .SamplerCP4
using .SamplerSurfaceCpx

# Parse command-line arguments
n_positive = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 2000
n_negative = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 2000

println("\n" * "="^60)
println("UNIFORM SAMPLING ON TRIPLE-ROOT SURFACE")
println("Using Crofton weights and CP² slicing")
println("="^60)

# Set random seed for reproducibility
Random.seed!(42)

# Create surface equations E1, E2
println("\nInitializing surface equations...")
H1, H2 = PolyEval.create_surface_equations()
println("  E1 = a2² - 3a1a3 + 12a0a4")
println("  E2 = a1a2a3 - 9a0a3² - 9a1²a4 + 32a0a2a4")

# Generate positive samples (on surface) using UNIFORM sampling
println("\n1. Generating $n_positive positive samples (UNIFORM on surface)...")
println("   Method: Random CP² slices + Crofton weights")
println("   This ensures uniformity w.r.t. Fubini-Study measure")
println()

A_positive = SamplerSurfaceCpx.sample_surface(
    n_positive, 
    Utils.random_C3_subspace,
    SliceSystem.ambient_point,
    SliceSystem.solve_slice,
    PolyEval.heval,
    PolyEval.hgrad,
    H1, H2;
    tol=1e-10,
    verbose=true
)

println("   ✓ Generated $(size(A_positive, 2)) positive samples")

# Convert to real-valued features (real and imaginary parts)
n_generated = size(A_positive, 2)
positive_matrix = zeros(n_generated, 10)  # 5 complex coords = 10 real features
for i in 1:n_generated
    pt = A_positive[:, i]  # Column vector
    positive_matrix[i, 1:5] = real.(pt)
    positive_matrix[i, 6:10] = imag.(pt)
end

# Generate negative samples (uniform random points in CP^4)
println("\n2. Generating $n_negative negative samples (UNIFORM in CP⁴)...")
println("   Method: Fubini-Study measure (ℓ2-normalized random)")
A_negative = SamplerCP4.sample_cp4(n_negative)

negative_matrix = zeros(n_negative, 10)
for i in 1:n_negative
    pt = A_negative[:, i]  # Column vector
    negative_matrix[i, 1:5] = real.(pt)
    negative_matrix[i, 6:10] = imag.(pt)
end
println("   ✓ Generated $n_negative negative samples")

# Combine and create labels
println("\n3. Creating combined dataset...")
X = vcat(positive_matrix, negative_matrix)
y = vcat(ones(n_generated), zeros(n_negative))

# Shuffle
println("   Shuffling dataset...")
n = size(X, 1)
indices = randperm(n)
X_shuffled = X[indices, :]
y_shuffled = y[indices]

# Save to CSV
println("\n4. Saving to files...")
writedlm("data/X_train.csv", X_shuffled, ',')
writedlm("data/y_train.csv", y_shuffled, ',')
writedlm("data/positive_samples.csv", positive_matrix, ',')
writedlm("data/negative_samples.csv", negative_matrix, ',')

println("   Saved X_train.csv: $(size(X_shuffled))")
println("   Saved y_train.csv: $(length(y_shuffled))")
println("   Saved positive_samples.csv")
println("   Saved negative_samples.csv")
println("   Location: data/")

# Verify positive samples are actually on the surface
println("\n5. Verifying positive samples satisfy E1=0, E2=0...")
using StaticArrays

function verify_sample(pt_real, pt_imag, H1, H2)
    a = SVector{5,ComplexF64}(
        pt_real[1] + im*pt_imag[1],
        pt_real[2] + im*pt_imag[2],
        pt_real[3] + im*pt_imag[3],
        pt_real[4] + im*pt_imag[4],
        pt_real[5] + im*pt_imag[5]
    )
    
    E1_val = PolyEval.heval(H1, a)
    E2_val = PolyEval.heval(H2, a)
    
    return abs(E1_val), abs(E2_val)
end

n_check = min(10, n_generated)
residuals = [verify_sample(positive_matrix[i, 1:5], positive_matrix[i, 6:10], H1, H2) for i in 1:n_check]
max_e1 = maximum([r[1] for r in residuals])
max_e2 = maximum([r[2] for r in residuals])
avg_e1 = sum([r[1] for r in residuals]) / n_check
avg_e2 = sum([r[2] for r in residuals]) / n_check

println("   Checked $n_check samples:")
println("   Max |E1| residual: $(max_e1)")
println("   Max |E2| residual: $(max_e2)")
println("   Avg |E1| residual: $(avg_e1)")
println("   Avg |E2| residual: $(avg_e2)")
if max_e1 < 1e-8 && max_e2 < 1e-8
    println("   ✓ All samples verified on surface (residuals < 10⁻⁸)")
else
    println("   ⚠ Some residuals larger than expected")
end

println("\n" * "="^60)
println("DATASET GENERATION COMPLETE")
println("="^60)
println("\nDataset summary:")
println("  Total samples: $(n_generated + n_negative)")
println("  Positive (UNIFORM on surface): $n_generated")
println("  Negative (UNIFORM in CP⁴): $n_negative")
println("  Features per sample: 10 (5 real + 5 imaginary parts)")
println("  Output location: data/")
println("\nSampling methodology:")
println("  ✓ Positive: Crofton-weighted CP² slicing (README §5-7)")
println("  ✓ Negative: Fubini-Study measure on CP⁴ (README §4)")
println("  ✓ Both methods are unitary-invariant and theoretically sound")
println("="^60)
