module SamplerSurfaceCpx
using Random, LinearAlgebra, StaticArrays, StatsBase

"Draw M i.i.d. uniform points on the triple-root surface in ℂP⁴ using Crofton weights."
function sample_surface(M::Int, random_C3_subspace_func, ambient_point_func, 
                        solve_slice_func, heval_func, hgrad_func, H1, H2; 
                        rng=Random.default_rng(), tol=1e-10, max_attempts=10000, 
                        verbose=true)
    A = Matrix{ComplexF64}(undef, 5, M)
    cnt = 0
    attempts = 0
    failed_slices = 0
    
    if verbose
        println("Sampling $M points uniformly on the surface using Crofton weights...")
    end
    
    while cnt < M && attempts < max_attempts
        attempts += 1
        
        # Random CP^2 slice (unitary-invariant)
        B = random_C3_subspace_func(; rng=rng)
        
        # Solve H1=0, H2=0 on this slice
        XY, wts = solve_slice_func(B, heval_func, hgrad_func, H1, H2; tol=tol, show_progress=false)
        
        K = size(XY, 2)
        if K == 0 || sum(wts) == 0
            failed_slices += 1
            continue
        end
        
        # Select ONE point from this slice with probability ∝ Crofton weight
        # This is the KEY to uniform sampling
        p = wts ./ sum(wts)
        k = sample(rng, 1:K, Weights(p))
        
        x, y = XY[1, k], XY[2, k]
        a = ambient_point_func(B, x, y)
        a_normalized = a ./ norm(a)      # Normalize to unit ℓ2 norm (returns new SVector)
        
        cnt += 1
        A[:, cnt] = a_normalized
        
        if verbose && cnt % 100 == 0
            println("  Progress: $cnt/$M samples ($(failed_slices) failed slices)")
        end
    end
    
    if cnt < M
        @warn "Only generated $cnt/$M samples after $attempts attempts"
    end
    
    if verbose
        success_rate = 100 * (attempts - failed_slices) / attempts
        println("  Completed: $cnt/$M samples")
        println("  Success rate: $(round(success_rate, digits=1))% of slices had solutions")
    end
    
    return A[:, 1:cnt]  # Return what we got
end

export sample_surface
end
