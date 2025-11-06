module Utils
using LinearAlgebra, Random

"Return 5×3 matrix with orthonormal columns spanning a random 3-plane W (random CP^2 slice)."
function random_C3_subspace(; rng=Random.default_rng())
    Z = randn(rng, ComplexF64, 5, 3)
    Q, _ = qr(Z)  # complex QR
    return Matrix(Q[:, 1:3])
end

export random_C3_subspace
end
