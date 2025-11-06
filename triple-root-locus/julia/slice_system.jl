module SliceSystem
using HomotopyContinuation, StaticArrays, LinearAlgebra

@var x y

"Ambient representative a(x,y) = B * [x, y, 1]. B is 5×3 complex."
@inline function ambient_point(B::AbstractMatrix{ComplexF64}, x::ComplexF64, y::ComplexF64)
    return SVector{5,ComplexF64}(B * SVector{3,ComplexF64}(x, y, 1+0im))
end

"Crofton weight w = 1/|det J|, where J_ij = ∂H_i/∂(var_j) at (x*,y*)."
function crofton_weight(B::AbstractMatrix{ComplexF64}, heval_func, hgrad_func, 
                        H1, H2, x::ComplexF64, y::ComplexF64)
    a = ambient_point(B, x, y)
    g1 = hgrad_func(H1, a)
    g2 = hgrad_func(H2, a)
    dax = SVector{5,ComplexF64}(B[:,1])
    day = SVector{5,ComplexF64}(B[:,2])
    J11 = dot(g1, dax);  J12 = dot(g1, day)
    J21 = dot(g2, dax);  J22 = dot(g2, day)
    detJ = J11*J22 - J12*J21
    return 1.0 / max(abs(detJ), 1e-14)   # clip to avoid division by zero
end

"Solve the slice; return solutions (x,y) and corresponding Crofton weights."
function solve_slice(B::AbstractMatrix{ComplexF64}, heval_func, hgrad_func, 
                     H1, H2; tol=1e-10, show_progress=false)
    # Define system F(x,y) = (H1(a(x,y)), H2(a(x,y)))
    @var x y
    a_vars = B * [x, y, 1]
    
    # Build polynomial system from the sparse representation
    E1_expr = sum(c * prod(a_vars[i]^e[i] for i in 1:5) for (c, e) in zip(H1.coeffs, H1.exps))
    E2_expr = sum(c * prod(a_vars[i]^e[i] for i in 1:5) for (c, e) in zip(H2.coeffs, H2.exps))
    
    F = System([E1_expr, E2_expr], variables=[x, y])
    
    # Solve
    result = solve(F; show_progress=show_progress)
    
    xy = ComplexF64[]
    wts = Float64[]
    
    for s in solutions(result)
        x_val, y_val = s[1], s[2]
        a = ambient_point(B, x_val, y_val)
        
        # Check residuals
        r1 = abs(heval_func(H1, a))
        r2 = abs(heval_func(H2, a))
        
        if r1 ≤ tol && r2 ≤ tol
            push!(xy, x_val)
            push!(xy, y_val)
            w = crofton_weight(B, heval_func, hgrad_func, H1, H2, x_val, y_val)
            push!(wts, w)
        end
    end
    
    if length(xy) > 0
        return reshape(xy, 2, :), wts
    else
        return zeros(ComplexF64, 2, 0), Float64[]
    end
end

export ambient_point, crofton_weight, solve_slice
end
