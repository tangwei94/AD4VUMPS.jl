using TensorKit, TensorOperations, KrylovKit
using ChainRules, ChainRulesCore, Zygote
using Revise
using MPSTransferMatrix

sp1 = ℂ^6;
sp2 = ℂ^2;

A = TensorMap(rand, ComplexF64, sp1*sp2, sp1);
M = TensorMap(rand, ComplexF64, sp2*sp2, sp2*sp2);

AC = TensorMap(rand, ComplexF64, sp1*sp2, sp1);
C = TensorMap(rand, ComplexF64, sp1, sp1);

AL, AR, conv_meas = mps_update(AC, C)

function _F(AC, C)
    AL, AR, conv_meas = mps_update(AC, C)
    return norm(tr(AL'*AR))/norm(AL)/norm(AR) 
end

function _F1(C)
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

    return norm(tr(AL'*AR))/norm(AL)/norm(AR) 
end

_F1(C)
_F1'(C)


Q = similar(AL)
randomize!(Q)
QC = similar(C)
randomize!(QC)

function _F(M)
    TM_L = MPSMPOMPSTransferMatrix(AL, M, AL, false)
    TM_R = MPSMPOMPSTransferMatrix(AR, M, AR, false)

    EL = left_env(TM_L)
    ER = right_env(TM_R)

    ER_permuted = permute(ER, (3, 2), (1, ))
    EL_permuted = permute(EL', (3, 2), (1, ))

    AC_map = MPSMPOMPSTransferMatrix(EL_permuted, M, ER_permuted, false)
    AC_permuted = right_env(AC_map) 
    AC = permute(AC_permuted, (3, 2), (1, ))

    return norm(Q' * AC) / norm(AC)
end

function _FC(M)
    TM_L = MPSMPOMPSTransferMatrix(AL, M, AL, false)
    TM_R = MPSMPOMPSTransferMatrix(AR, M, AR, false)

    EL = left_env(TM_L)
    ER = right_env(TM_R)

    C_map = MPSMPSTransferMatrix(EL', ER, false)
    C = left_env(C_map) 

    return norm(QC' * C) / norm(C)
end

function tensor_square_ising(β::Real)
    t = TensorMap(ComplexF64[exp(β) exp(-β); exp(-β) exp(β)], ℂ^2, ℂ^2)
    sqrt_t = sqrt(t)
    δ = TensorMap(zeros, ComplexF64, ℂ^2*ℂ^2, ℂ^2*ℂ^2)

    δ[1, 1, 1, 1] = 1
    δ[2, 2, 2, 2] = 1 

    @tensor T[-1 -2 ; -3 -4] := sqrt_t[-1; 1] * sqrt_t[-2; 2] * sqrt_t[3; -3] * sqrt_t[4; -4] * δ[1 2; 3 4]
    return T
end
βc = asinh(1) / 2
M = tensor_square_ising(βc)
A = TensorMap(rand, ComplexF64, ℂ^4*ℂ^2, ℂ^4) 
vumps(A, M)





