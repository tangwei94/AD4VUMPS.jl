function gauge_fixing(AC1, AC2, S)

    function _f(U)
        #Cost1 = norm(U * S - S * U)^2
        a = U * S - S * U
        Cost1 = real(dot(a, a))
        @tensor cost2_a[-1 -2; -3] := AC1[1 -2; -3] * U[-1; 1]
        @tensor cost2_b[-1 -2; -3] := AC2[-1 -2; 1] * U[1; -3]
        b = cost2_a - cost2_b
        Cost2 = real(dot(b, b))

        return sqrt(Cost1 + Cost2)
        #Cost1 = norm(U * S - S * U)^2
        #@tensor cost2_a[-1 -2; -3] := AC1[1 -2; -3] * U[-1; 1]
        #@tensor cost2_b[-1 -2; -3] := AC2[-1 -2; 1] * U[1; -3]
        #Cost2 = 0#norm(cost2_a - cost2_b)^2

        #return sqrt(Cost1 + Cost2)
    end

    function _fg(U)
        f, g = withgradient(_f, U)
        dU = Unitary.project!(g[1], U)
        return f, dU
    end

    M = TensorMap(rand, ComplexF64, space(S))
    U, _ = leftorth(M)
    optalg_LBFGS = LBFGS(;gradtol=1e-12, maxiter=250, verbosity=2)
    U, fvalue, grad, _, _ = optimize(_fg, U, optalg_LBFGS; 
                                             transport! = Unitary.transport!,
                                             retract = Unitary.retract,
                                             inner = Unitary.inner,
                                             scale! = Unitary.rmul!,
                                             add! =(U, gU, α) -> axpy!(α, gU, U))

    if norm(grad) > 1e-12
        @warn "Gauge fixing did not converge, $(fvalue), $(norm(grad))"
    end
    return U
end

function gauge_fixing(AC1, C1, AC2, C2)
    U1, S1, V1 = tsvd(C1)
    U2, S2, V2 = tsvd(C2)

    (norm(S1 - S2) > 1e-10) && (@warn "your VUMPS has not converged yet, the gauge fixing may not work properly")

    @tensor AC1_updated[-1 -2; -3] := AC1[1 -2; 2] * U1'[-1; 1] * V1'[2; -3]
    @tensor AC2_updated[-1 -2; -3] := AC2[1 -2; 2] * U2'[-1; 1] * V2'[2; -3]

    @show norm(S1 - S2), norm(AC1_updated - AC2_updated)
    return gauge_fixing(AC1_updated, AC2_updated, S1)
end
@non_differentiable gauge_fixing(args...)