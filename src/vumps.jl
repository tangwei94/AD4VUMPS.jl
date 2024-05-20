function mps_update(AC::MPSTensor, C::MPSBondTensor)
    UAC_l, PAC_l = leftorth(AC; alg = QRpos())
    UC_l, PC_l = leftorth(C; alg = QRpos())

    PAC_r, UAC_r = rightorth(permute(AC, ((1,), (2, 3))); alg = LQpos())
    PC_r, UC_r = rightorth(C; alg=LQpos())

    AL = UAC_l * UC_l'
    AR = permute(UC_r' * UAC_r, ((1, 2), (3,)))

    # check AC - AL * C and AC - C * AR
    conv_meas = ignore_derivatives() do
        ϵL = norm(PAC_l - PC_l) 
        ϵR = norm(PAC_r - PC_r)
        conv_meas = max(ϵL, ϵR)
        return conv_meas
    end

    return AL, AR, conv_meas
end

function vumps_update(AL::MPSTensor, AR::MPSTensor, T::MPOTensor)

    TM_L = MPSMPOMPSTransferMatrix(AL, T, AL)
    TM_R = MPSMPOMPSTransferMatrix(AR, T, AR)

    EL = left_env(TM_L)
    ER = right_env(TM_R)

    # AC map
    ER_permuted = permute(ER, ((3, 2), (1, )))
    EL_permuted = permute(EL', ((3, 2), (1, )))

    AC_map = MPSMPOMPSTransferMatrix(EL_permuted, T, ER_permuted)
    AC_permuted = right_env(AC_map) 
    AC = permute(AC_permuted, ((3, 2), (1, )))

    # C map
    C_map = MPSMPSTransferMatrix(EL', ER)
    C = left_env(C_map) 

    return AC, C
end

function vumps(A::MPSTensor, T::MPOTensor; maxiter=500, tol=1e-12)
    # TODO.: canonical form conversion
    sp = domain(A)[1]
    C = TensorMap(rand, ComplexF64, sp, sp)
    AL, AR = mps_update(A, C)
    conv_meas = 999
    ix = 0
    AC, C = vumps_update(AL, AR, T)
    while conv_meas > tol && ix < maxiter
        ix += 1
        AC1, C1 = vumps_update(AL, AR, T)
        AL, AR, conv_meas = mps_update(AC1, C1)
        AC, C = AC1, C1
        print(ix, ' ', conv_meas, "     \r")
    end
    print("\n")
    return AL, AR, AC, C
end

@non_differentiable vumps(::MPSTensor, ::MPOTensor) 

function vumps_for_ad(T::MPOTensor; AL::MPSTensor, AR::MPSTensor, AC::MPSTensor, C::MPSBondTensor, maxiter=100)
    AL1 = ignore_derivatives(AL) 
    AR1 = ignore_derivatives(AR) 
    AC1 = ignore_derivatives(AC) 
    C1 = ignore_derivatives(C) 
    for _ in 1:maxiter
        AC1, C1 = vumps_update(AL1, AR1, T)
        AL1, AR1, _ = mps_update(AC1, C1)
    end
    return AL1, AR1, AC1, C1
end