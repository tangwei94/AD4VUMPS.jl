using LinearAlgebra
using TensorKit, TensorOperations, KrylovKit
using ChainRules, ChainRulesCore, Zygote, OptimKit
using MPSKit
using TensorKitManifolds
using Revise
using AD4VUMPS

# testing the arnoldi method for the vumps pushback
T = tensor_square_ising(asinh(1) / 2) 
A = TensorMap(rand, ComplexF64, ℂ^4*ℂ^2, ℂ^4) 

O = tensor_square_ising_O(asinh(1) / 2 / 2)

AL, AR = vumps(T; A=A, verbosity=1)
AL1, AR1 = deepcopy(AL), deepcopy(AR)

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

vumps_iteration_vjp = pullback(AD4VUMPS.gauge_fixed_vumps_iteration, AL, AR, T)[2]

X1 = vumps_iteration_vjp(∂ALAR);
    
function vjp_ALAR_ALAR(X)
    res = vumps_iteration_vjp((X[1], X[2]))
    return (res[1], res[2])
end
vjp_ALAR_T(X) = vumps_iteration_vjp((X[1], X[2]))[3]

X1 = vjp_ALAR_ALAR([∂AL, ∂AR]) 
Y1 = (X1[1], X1[2], 1.0+0.0im)

AR_perm = permute(AR, ((1, ), (2, 3)))'
y = TensorMap(rand, ComplexF64, (ℂ^2)'*(ℂ^4), ℂ^4)

AR_perm

function f_map(Y)
    TensorKitManifolds.Stiefel.project!(Y[1], AL)

    AR_perm = permute(AR, ((1, ), (2, 3)))'
    Y2_perm = permute(Y[2], ((1, ), (2, 3)))'
    TensorKitManifolds.Stiefel.project!(Y2_perm, AR_perm)

    Yx = vjp_ALAR_ALAR([Y[1], permute(Y2_perm', ((1, 2), (3, )))])

    TensorKitManifolds.Stiefel.project!(Yx[1], AL)
    Yx2_perm = permute(Yx[2], ((1, ), (2, 3)))'
    TensorKitManifolds.Stiefel.project!(Yx2_perm, AR_perm)

    #TensorKitManifolds.Stiefel.project!(Yx[1], AL)
    return (Yx[1] + X1[1], permute(Yx2_perm', ((1, 2), (3, ))) + X1[2], Y[3])
end
Y0 = (zero(Y1[1]), zero(Y1[2]), 0.0+0.0im)
M = zeros(ComplexF64, 65, 65)
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

vals, vecs, info = eigsolve(f_map, Y1, 2, :LR; tol=1e-6)
@show vals
for ix in eachindex(vals)
    @show norm(vals[ix]), norm(vecs[ix][end])
end
vals, vecs, info = eigsolve(f_map, Y1, 10, :LR; tol=1e-6) 
@show vals
for ix in 1:10
    @show norm(vals[ix]), norm(vecs[ix][end])
end