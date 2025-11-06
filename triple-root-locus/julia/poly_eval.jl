module PolyEval
using StaticArrays

struct SparsePoly
    coeffs::Vector{ComplexF64}
    exps::Vector{SVector{5,Int}}
end

"""
Load polynomial from explicit coefficients and exponents.
For the triple-root surface:
E1 = a2^2 - 3*a1*a3 + 12*a0*a4
E2 = a1*a2*a3 - 9*a0*a3^2 - 9*a1^2*a4 + 32*a0*a2*a4
"""
function create_surface_equations()
    # E1 = a2^2 - 3*a1*a3 + 12*a0*a4
    E1_coeffs = [ComplexF64(1, 0), ComplexF64(-3, 0), ComplexF64(12, 0)]
    E1_exps = [SVector{5,Int}(0,0,2,0,0), SVector{5,Int}(0,1,0,1,0), SVector{5,Int}(1,0,0,0,1)]
    
    # E2 = a1*a2*a3 - 9*a0*a3^2 - 9*a1^2*a4 + 32*a0*a2*a4
    E2_coeffs = [ComplexF64(1, 0), ComplexF64(-9, 0), ComplexF64(-9, 0), ComplexF64(32, 0)]
    E2_exps = [SVector{5,Int}(0,1,1,1,0), SVector{5,Int}(1,0,0,2,0), 
               SVector{5,Int}(0,2,0,0,1), SVector{5,Int}(1,0,1,0,1)]
    
    E1 = SparsePoly(E1_coeffs, E1_exps)
    E2 = SparsePoly(E2_coeffs, E2_exps)
    
    return E1, E2
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

"Numerical complex gradient via central differences."
function hgrad(p::SparsePoly, a::SVector{5,ComplexF64}; eps=1e-7)
    g = zeros(ComplexF64, 5)
    for k in 1:5
        a_plus = SVector{5,ComplexF64}(i == k ? a[i] + eps : a[i] for i in 1:5)
        a_minus = SVector{5,ComplexF64}(i == k ? a[i] - eps : a[i] for i in 1:5)
        g[k] = (heval(p, a_plus) - heval(p, a_minus)) / (2eps)
    end
    return SVector{5,ComplexF64}(g)
end

export SparsePoly, create_surface_equations, heval, hgrad
end
