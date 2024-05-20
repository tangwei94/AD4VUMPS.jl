using TensorKit, TensorOperations, KrylovKit
using ChainRules, ChainRulesCore, Zygote
using MPSKit
using Revise
using AD4VUMPS

sp1, sp2 = ℂ^4, ℂ^2 
βc = asinh(1) / 2 
T = tensor_square_ising(βc) 
A = TensorMap(rand, ComplexF64, sp1*sp2, sp1) 
O = tensor_square_ising_O(βc) 

function _F1(T)
    AL, AR, AC, C = vumps(A, T)
    ψ, _ = leading_boundary(InfiniteMPS([A]), DenseMPO([T]), VUMPS())
    @show norm(dot(ψ, InfiniteMPS([AL])))
    @show norm(dot(ψ, InfiniteMPS([AR])))
    AL1, AR1, AC1, C1 = vumps_for_ad(T; AL=AL, AR=AR, AC=AC, C=C)
    TM = MPSMPOMPSTransferMatrix(AL1, T, AL1, false)
    EL = left_env(TM)
    ER = right_env(TM)

    @tensor a = EL[4; 1 2] * AL1[1 3; 6] * O[2 5; 3 8] * conj(AL1[4 5; 7]) * ER[6 8; 7]
    @tensor b = EL[4; 1 2] * AL1[1 3; 6] * T[2 5; 3 8] * conj(AL1[4 5; 7]) * ER[6 8; 7]
    return real(a/b)
end

s = random_real_symmetric_tensor(2) 
_F1(T) 

# forward + finite difference is already unstable 
let sX = random_real_symmetric_tensor(2), α = 1e-6, _F = _F1, X = T
    ap2 = _F(X + 2*α * sX)
    ap1 = _F(X + α * sX)
    am1 = _F(X - α * sX)
    am2 = _F(X - 2*α * sX)
    @show ap2, ap1, am1, am2
    ∂α1 = (-_F(X + 2*α * sX) + 8*_F(X + α * sX) - 8*_F(X - α * sX) + _F(X - 2*α * sX)) / (12 * α)
    ap2 = _F(X + 2*α * sX)
    ap1 = _F(X + α * sX)
    am1 = _F(X - α * sX)
    am2 = _F(X - 2*α * sX)
    @show ap2, ap1, am1, am2
    ∂α2 = (-_F(X + 2*α * sX) + 8*_F(X + α * sX) - 8*_F(X - α * sX) + _F(X - 2*α * sX)) / (12 * α)
    @show ∂α1, ∂α2  
    return norm(∂α1 - ∂α2)/ norm(∂α1)
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