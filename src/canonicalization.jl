function right_canonical_QR(A::MPSTensor; tol::Float64=1e-15, maxiter=200, enable_warning=false)

    L, Q = rightorth(permute(A, ((1, ), (2, 3))))
    AR = permute(Q, ((1, 2), (3, )))
    L = L / norm(L)
    δ = norm(L - id(domain(L)[1])) 
    L0 = L

    ix= 0
    while δ > tol && ix < maxiter
        if ix >= maxiter ÷ 10 && ix % 10 == 0 
            lop = MPSMPSTransferMatrix(A, AR)
            L = right_env(lop)' # TODO. tol = max(tol, δ/10)
        end

        L, Q = rightorth(permute(A * L, ((1, ), (2, 3))))
        AR = permute(Q, ((1, 2), (3, )))
        L = L / norm(L)

        δ = norm(L-L0)
        L0 = L

        ix += 1
    end
    
    enable_warning && δ > tol && @warn "right_canonical_QR failed to converge. δ: $δ , tol: $tol"

    return AR, L0'
end

function left_canonical_QR(A::TensorMap{ComplexSpace, 2, 1}; tol::Float64=1e-15, maxiter=200, enable_warning=false)

    AL, R = leftorth(A)
    R = R / norm(R)
    δ = norm(R - id(domain(R)[1])) 
    R0 = R

    ix = 0
    while δ > tol && ix < maxiter
        if ix >= maxiter ÷ 10 && ix % 10 == 0 
            lop = MPSMPSTransferMatrix(A, AL) # TODO. tol = max(tol, δ/10)
            R = left_env(lop)
        end

        @tensor A_tmp[-1, -2; -3] := R[-1, 1] * A[1, -2, -3]
        AL, R = leftorth(A_tmp) # TODO. tol = max(tol, δ/10)
        R = R / norm(R)

        δ = norm(R - R0)
        R0 = R

        ix += 1
    end

    #println(ix, " iterations")
    enable_warning && δ > tol && @warn "left_canonical_QR failed to converge. δ: $δ , tol: $tol"

    return AL, R0
end