# Example Usage Script for Uniform Sampling
# This demonstrates how to use the uniform_sampling module

include("uniform_sampling.jl")

println("\n" * "="^60)
println("EXAMPLE: Generating datasets for machine learning")
println("="^60)

# Example 1: Generate a dataset of 100 complex samples
println("\n1. Generating 100 complex samples...")
complex_samples = sample_k_complex(100, verbose=true)
println("   Generated $(length(complex_samples)) complex samples")
println("   Sample dimension: $(length(complex_samples[1])) (projective coordinates)")

# Example 2: Generate 50 real samples
println("\n2. Generating 50 real samples...")
real_samples, n_slices_needed = sample_k_real_points(50, verbose=true)
println("   Generated $(length(real_samples)) real samples")
println("   Needed $n_slices_needed random slices")

# Example 3: Save samples to a file
println("\n3. Saving samples to files...")

# Save complex samples
using DelimitedFiles
complex_matrix = hcat([real.(s) for s in complex_samples]...)'  # Real parts only for demo
writedlm("complex_samples_real_parts.csv", complex_matrix, ',')
println("   Saved complex sample real parts to: complex_samples_real_parts.csv")

# Save real samples
real_matrix = hcat(real_samples...)'
writedlm("real_samples.csv", real_matrix, ',')
println("   Saved real samples to: real_samples.csv")

# Example 4: Verify samples are on the surface
println("\n4. Verifying samples satisfy the surface equations...")

function verify_on_surface(pt; tol=1e-6)
    # Evaluate E1 and E2
    a0_val, a1_val, a2_val, a3_val, a4_val = pt
    
    # E1 = a2^2 - 3*a1*a3 + 12*a0*a4
    E1_val = a2_val^2 - 3*a1_val*a3_val + 12*a0_val*a4_val
    # E2 = a1*a2*a3 - 9*a0*a3^2 - 9*a1^2*a4 + 32*a0*a2*a4
    E2_val = a1_val*a2_val*a3_val - 9*a0_val*a3_val^2 - 9*a1_val^2*a4_val + 32*a0_val*a2_val*a4_val
    
    return abs(E1_val) < tol && abs(E2_val) < tol
end

# Check first 10 real samples
function count_verified_samples(samples, n_check)
    count = 0
    for i in 1:min(n_check, length(samples))
        if verify_on_surface(samples[i], tol=1e-5)
            count += 1
        end
    end
    return count
end

n_verified = count_verified_samples(real_samples, 10)
println("   Verified $n_verified out of 10 samples lie on the surface (within tolerance)")

# Example 5: Statistics on real vs complex
println("\n5. Statistics:")
println("   Average number of real solutions per slice: ", 
        length(real_samples) / n_slices_needed)
println("   Expected: varies from 0 to 6 per slice")
println("   Surface degree: 6 (should get exactly 6 complex solutions per slice)")

println("\n" * "="^60)
println("Examples completed!")
println("="^60)
