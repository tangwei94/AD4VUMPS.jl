using LinearAlgebra
using TensorKit, TensorOperations, KrylovKit
using ChainRules, ChainRulesCore, Zygote, OptimKit
using TensorKitManifolds
using Revise
using AD4VUMPS

# testing the arnoldi method for the vumps pushback

function tensor_square_ising(β::Real) # tensor for classical Ising model
    t = TensorMap(ComplexF64[exp(β) exp(-β); exp(-β) exp(β)], ℂ^2, ℂ^2)
    sqrt_t = sqrt(t)
    δ = zeros(ComplexF64, ℂ^2*ℂ^2, ℂ^2*ℂ^2)

    δ[1, 1, 1, 1] = 1
    δ[2, 2, 2, 2] = 1 

    @tensor T[-1 -2 ; -3 -4] := sqrt_t[-1; 1] * sqrt_t[-2; 2] * sqrt_t[3; -3] * sqrt_t[4; -4] * δ[1 2; 3 4]
    return T
end
function tensor_square_ising_O(β::Real) # tensor for an "observable"
    t = TensorMap(ComplexF64[exp(β) exp(-β); exp(-β) exp(β)], ℂ^2, ℂ^2)
    sqrt_t = sqrt(t)
    δ = zeros(ComplexF64, ℂ^2*ℂ^2, ℂ^2*ℂ^2)

    δ.data .= 1

    @tensor T[-1 -2 ; -3 -4] := sqrt_t[-1; 1] * sqrt_t[-2; 2] * sqrt_t[3; -3] * sqrt_t[4; -4] * δ[1 2; 3 4]
    return T
end

# tensor 
T = tensor_square_ising(asinh(1) / 2) 
# "observable" tensor
O = tensor_square_ising_O(asinh(1) / 2 / 2)

# MPS local tensor 
A = rand(ComplexF64, ℂ^4*ℂ^2, ℂ^4) 
# VUMPS -> fixed-point AL, AR
AL, AR = vumps(T; A=A, verbosity=1)
# cost function 
function _F1(AL1, AR1)
    TM = MPSMPOMPSTransferMatrix(AL1, T, AL1)
    
    EL = left_env(TM)
    ER = right_env(TM)

    @tensor a = EL[4; 1 2] * AL1[1 3; 6] * O[2 5; 3 8] * conj(AL1[4 5; 7]) * ER[6 8; 7]
    @tensor b = EL[4; 1 2] * AL1[1 3; 6] * T[2 5; 3 8] * conj(AL1[4 5; 7]) * ER[6 8; 7]
    return real(a/b)
end

# backward of the gauge-fixed VUMPS iteration
vumps_iteration_vjp = pullback(AD4VUMPS.gauge_fixed_vumps_iteration, AL, AR, T)[2]
# ∂ALAR -> ∂AL∂AR
function vjp_ALAR_ALAR(X)
    res = vumps_iteration_vjp((X[1], X[2]))
    return [res[1], res[2]]
end
# ∂ALAR -> ∂T
vjp_ALAR_T(X) = vumps_iteration_vjp((X[1], X[2]))[3]

# adjoint of AL, AR
_, ∂ALAR = withgradient(_F1, AL, AR)
∂AL, ∂AR = ∂ALAR

# one more vumps iteration without gauge fixing. AL1 should be the same as AL up to a gauge transformation
AL1, AR1 = AD4VUMPS.ordinary_vumps_iteration(AL, AR, T)  

X1 = vumps_iteration_vjp(∂ALAR);

project_dAL = AD4VUMPS.project_dAL
project_dAR = AD4VUMPS.project_dAR
    
∂AL1 = deepcopy(∂AL)
∂AL2 = deepcopy(∂AL)
a = project_dAL(∂AL1, AL, :Stiefel)
b = project_dAL(∂AL2, AL1, :Stiefel)

∂AL3 = deepcopy(∂AL)
∂AL4 = deepcopy(∂AL)
c = project_dAL(∂AL3, AL, :Grassmann)
d = project_dAL(∂AL4, AL1, :Grassmann)

@show a - b |> norm
@show c - d |> norm
@show a - c |> norm
@show b - d |> norm
