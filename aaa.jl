using LinearAlgebra
using TensorKit, TensorOperations, KrylovKit
using ChainRules, ChainRulesCore, Zygote, OptimKit
using MPSKit
using Revise
using AD4VUMPS

# testing the arnoldi method for the vumps pushback

T = tensor_square_ising(asinh(1) / 2)
A = TensorMap(rand, ComplexF64, ℂ^4*ℂ^2, ℂ^4) 

O = tensor_square_ising_O(asinh(1) / 2 / 2)

AL, AR = vumps(T; A=A, verbosity=1)

_, vumps_iteration_vjp = pullback(AD4VUMPS.gauge_fixed_vumps_iteration, AL, AR, T)

function vjp_ALAR_ALAR(X)
    res = vumps_iteration_vjp((X[1], X[2]))
    return [res[1], res[2]]
end
vjp_ALAR_T(X) = vumps_iteration_vjp((X[1], X[2]))[3]

function _F1(AL1, AR1)
    TM = MPSMPOMPSTransferMatrix(AL1, T, AL1)
    EL = left_env(TM)
    ER = right_env(TM)

    @tensor a = EL[4; 1 2] * AL1[1 3; 6] * O[2 5; 3 8] * conj(AL1[4 5; 7]) * ER[6 8; 7]
    @tensor b = EL[4; 1 2] * AL1[1 3; 6] * T[2 5; 3 8] * conj(AL1[4 5; 7]) * ER[6 8; 7]
    return real(a/b)
end

_, ∂ALAR = withgradient(_F1, AL, AR)
∂AL, ∂AR = ∂ALAR

X1 = vumps_iteration_vjp(∂ALAR)
    
function vjp_ALAR_ALAR(X)
    res = vumps_iteration_vjp((X[1], X[2]))
    return (res[1], res[2])
end
vjp_ALAR_T(X) = vumps_iteration_vjp((X[1], X[2]))[3]

X1 = vjp_ALAR_ALAR([∂AL, ∂AR]) 
Y1 = (X1[1], X1[2], 1.0+0.0im)

function f_map(Y)
    Yx = vjp_ALAR_ALAR([Y[1], Y[2]])
    return (Yx[1] + X1[1], Yx[2] + X1[2], Y[3])
end
vals, vecs, info = eigsolve(f_map, Y1, 2, :LM; tol=1e-6)
@show vals
vals, vecs, info = eigsolve(f_map, Y1, 10, :LM; tol=1e-6) # FIXME. taskfailedexception
@show vals
for ix in 1:10
    @show vals[ix], vecs[ix][end]
end