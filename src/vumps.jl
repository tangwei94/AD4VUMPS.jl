function mps_update(AC::MPSTensor, C::MPSBondTensor)
        UAC_l, PAC_l = leftorth(AC; alg=QRpos())
        UC_l, PC_l = leftorth(C; alg=QRpos())

        PAC_r, UAC_r = rightorth(permute(AC, (1,), (2,3)); alg=LQpos())
        PC_r, UC_r = rightorth(C; alg=LQpos())

        AL = UAC_l * UC_l'
        AR = permute(UC_r' * UAC_r, (1, 2), (3,))

        # check AC - AL * C and AC - C * AR
        conv_meas = ignore_derivatives() do
            ϵL = norm(PAC_l - PC_l) 
            ϵR = norm(PAC_r - PC_r)
            conv_meas = max(ϵL, ϵR)
            return conv_meas
        end

        return AL, AR, conv_meas
end

function vumps_update(AL, AR, T)
    TM_L = MPSMPOMPSTransferMatrix(AL, T, AL, false)
    TM_R = MPSMPOMPSTransferMatrix(AR, T, AR, false)

    EL = left_env(TM_L)
    ER = right_env(TM_R)

    # AC map
    ER_permuted = permute(ER, (3, 2), (1, ))
    EL_permuted = permute(EL', (3, 2), (1, ))

    AC_map = MPSMPOMPSTransferMatrix(EL_permuted, T, ER_permuted, false)
    AC_permuted = right_env(AC_map) 
    AC = permute(AC_permuted, (3, 2), (1, ))

    # C map
    C_map = MPSMPSTransferMatrix(EL', ER, false)
    C = left_env(C_map) 

    return AC, C
end

function vumps(A, T; maxiter=500, tol=1e-12)
    # TODO.: canonical form conversion
    sp = domain(A)[1]
    C = TensorMap(rand, ComplexF64, sp, sp)
    AL, AR = mps_update(A, C)

    conv_meas = 999
    ix = 0
    while conv_meas > tol
        ix += 1
        AC, C = vumps_update(AL, AR, T)
        AL, AR, conv_meas = mps_update(AC, C)
        println(ix, ' ', conv_meas)
    end
    return AL, AR
end
