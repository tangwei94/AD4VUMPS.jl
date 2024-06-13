using TensorKit, TensorOperations, KrylovKit, TensorKitManifolds
using ChainRules, ChainRulesCore, Zygote, OptimKit
using MPSKit
using Revise
using AD4VUMPS

s = random_real_symmetric_tensor(2) 
T = tensor_square_ising(asinh(1) / 2) 
A = TensorMap(rand, ComplexF64, ℂ^6*ℂ^2, ℂ^6) 
AL, AR = vumps(T; A=A, verbosity=0)
O = tensor_square_ising_O(asinh(1) / 2 / 2)

gauge_fixed_vumps_iteration = AD4VUMPS.gauge_fixed_vumps_iteration
AL1, AR1 = gauge_fixed_vumps_iteration(AL, AR, T)

res, vumps_iteration_vjp = pullback(gauge_fixed_vumps_iteration, AL, AR, T);
AL1, AR1 = res

function _F1(AL1, AR1, T)
    TM = MPSMPOMPSTransferMatrix(AL1, T, AL1)
    EL = left_env(TM)
    ER = right_env(TM)

    @tensor a = EL[4; 1 2] * AL1[1 3; 6] * O[2 5; 3 8] * conj(AL1[4 5; 7]) * ER[6 8; 7]
    @tensor b = EL[4; 1 2] * AL1[1 3; 6] * T[2 5; 3 8] * conj(AL1[4 5; 7]) * ER[6 8; 7]
    return real(a/b)
end
val, grad = withgradient(_F1, AL1, AR1, T)
∂AL = grad[1]
∂AR = grad[2]
isnothing(∂AR)
∂T = grad[3]

X1 

function vjp_ALAR_ALAR(X)
    res = vumps_iteration_vjp((X[1], X[2]))
    return [res[1], res[2]]
end
vjp_ALAR_T(X) = vumps_iteration_vjp((X[1], X[2]))[3]

X0 = [∂AL, ∂AR]
Xj = vjp_ALAR_ALAR(X0)
Xsum = Xj
for _ in 1:200
    Xj = vjp_ALAR_ALAR(Xj)
    Xsum += Xj
    @show norm(Xj)
end
(!isnothing(∂AL)) && (Xsum[1] += ∂AL)
(!isnothing(∂AR)) && (Xsum[2] += ∂AR)
vjp_ALAR_T(Xsum)


# linsolve is a bit unstable 
Xsum1, info = linsolve(vjp_ALAR_ALAR, X1, X1, 1, -1)
Xsum - Xsum1 |> norm

Xsum - vjp_ALAR_ALAR(Xsum) - X1 |> norm