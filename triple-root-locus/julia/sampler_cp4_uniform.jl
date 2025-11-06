module SamplerCP4
using LinearAlgebra, Random

"Draw N i.i.d. uniform points in ℂP⁴ (columns). Uses Fubini-Study measure."
function sample_cp4(N::Int; rng=Random.default_rng())
    X = Matrix{ComplexF64}(undef, 5, N)
    for j in 1:N
        v = randn(rng, ComplexF64, 5)
        v ./= norm(v)   # ℓ2-normalize for Fubini-Study measure
        X[:,j] = v
    end
    return X
end

export sample_cp4
end
