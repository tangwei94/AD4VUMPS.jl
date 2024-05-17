using TensorKit, TensorOperations, KrylovKit
using ChainRules, ChainRulesCore, Zygote
using Revise
using AD4VUMPS

sp1, sp2 = ℂ^4, ℂ^2
T = tensor_square_ising(asinh(1) / 2)
A = TensorMap(rand, ComplexF64, sp1*sp2, sp1) 
O = tensor_square_ising_O(asinh(1) / 2)

function _F1(T)
    AL, AR, AC, C = vumps(A, T)
    AL1, AR1, AC1, C1 = vumps_for_ad(T; AL=AL, AR=AR, AC=AC, C=C, maxiter=30)
    TM = MPSMPOMPSTransferMatrix(AL1, T, AL1, false)
    EL = left_env(TM)
    ER = right_env(TM)

    @tensor a = EL[4; 1 2] * AL1[1 3; 6] * O[2 5; 3 8] * conj(AL1[4 5; 7]) * ER[6 8; 7]
    @tensor b = EL[4; 1 2] * AL1[1 3; 6] * T[2 5; 3 8] * conj(AL1[4 5; 7]) * ER[6 8; 7]
    return real(a/b)
end

T = tensor_square_ising(asinh(1) / 2)
A = TensorMap(rand, ComplexF64, ℂ^4*ℂ^2, ℂ^4) 
AL, AR, AC, C = vumps(A, T)
O = tensor_square_ising_O(asinh(1) / 2)

function _F2(T)
    AL1, AR1, AC1, C1 = vumps_for_ad(T; AL=AL, AR=AR, AC=AC, C=C, maxiter=1)
    TM = MPSMPOMPSTransferMatrix(AL1, T, AL1, false)
    EL = left_env(TM)
    ER = right_env(TM)

    @tensor a = EL[4; 1 2] * AL1[1 3; 6] * O[2 5; 3 8] * conj(AL1[4 5; 7]) * ER[6 8; 7]
    @tensor b = EL[4; 1 2] * AL1[1 3; 6] * T[2 5; 3 8] * conj(AL1[4 5; 7]) * ER[6 8; 7]
    return real(a/b)
end
        
for ix in 1:100
    sX = random_real_symmetric_tensor(2)
    test_ADgrad(_F2, T; α=1e-4, tol=1e-4, sX=sX, num=1)
end