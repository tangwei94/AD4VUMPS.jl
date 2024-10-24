using LinearAlgebra
using TensorKit, TensorOperations, KrylovKit
using ChainRules, ChainRulesCore, Zygote, OptimKit
using TensorKitManifolds
using Revise
using AD4VUMPS

# testing the arnoldi method for the vumps pushback
T = tensor_square_ising(asinh(1) / 2) 
A = TensorMap(rand, ComplexF64, ℂ^4*ℂ^2, ℂ^4) 

O = tensor_square_ising_O(asinh(1) / 2 / 2)

AL, AR = vumps(T; A=A, verbosity=1)

vumps_iteration_vjp = pullback(AD4VUMPS.gauge_fixed_vumps_iteration, AL, AR, T)[2]
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

AL1, AR1 = AD4VUMPS.ordinary_vumps_iteration(AL, AR, T)

X1 = vumps_iteration_vjp(∂ALAR);

project_dAL = AD4VUMPS.project_dAL
project_dAR = AD4VUMPS.project_dAR
    
∂AL
∂AL1 = copy(∂AL)
∂AL2 = copy(∂AL)
project_dAL(∂AL1, AL, :Stiefel)
project_dAL(∂AL2, AL1, :Stiefel)

∂AL3 = copy(∂AL)
∂AL4 = copy(∂AL)
project_dAL!(∂AL3, AL, :Grassmann)
project_dAL!(∂AL4, AL1, :Grassmann)

AL - AL1 |> norm
∂AL1 - ∂AL2 |> norm
∂AL1 - ∂AL2 |> norm
∂AL3 - ∂AL4 |> norm
∂AL1 - ∂AL3 |> norm

vjp_ALAR_T(X) = vumps_iteration_vjp((X[1], X[2]))[3]

X1 = vjp_ALAR_ALAR([∂AL, ∂AR]) 
Y1 = (X1[1], X1[2], 1.0+0.0im)

function f_map(Y)
    Yx = vjp_ALAR_ALAR([Y[1], Y[2]])
    return (Yx[1] + Y[3] * X1[1], Yx[2] + Y[3] * X1[2], Y[3])
end
Y0 = (zero(Y1[1]), zero(Y1[2]), 0.0+0.0im)
M = zeros(ComplexF64, 65, 65);
for ix in 1:65
    Yi = deepcopy(Y0)
    if ix <= 32
        Yi[1][ix] = 1.0
    elseif ix <= 64
        Yi[2][ix - 32] = 1.0
    else
        Yi = (Y0[1], Y0[2], 1.0+0.0im)
    end

    Yo = f_map(Yi)
    TensorKitManifolds.Stiefel.project!(Yo[1], AL)
    M[:, ix] = [vec(Yo[1].data); vec(Yo[2].data); Yo[3]]
end
λs, vecs = eigen(M)
λs
vecs

vals, vecs, info = eigsolve(f_map, Y1, 1, :LM; tol=1e-6)
@show vals
vecs[1][end] 
for ix in eachindex(vals)
    @show norm(vals[ix]), norm(vecs[ix][end])
end
vals, vecs, info = eigsolve(f_map, Y1, 10, :LR; tol=1e-6) 
@show vals
for ix in 1:10
    @show norm(vals[ix]), norm(vecs[ix][end])
end