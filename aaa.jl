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

function tensor_square_ising(β::Real)
    t = TensorMap(ComplexF64[exp(β) exp(-β); exp(-β) exp(β)], ℂ^2, ℂ^2)
    sqrt_t = sqrt(t)
    δ = TensorMap(zeros, ComplexF64, ℂ^2*ℂ^2, ℂ^2*ℂ^2)

    δ[1, 1, 1, 1] = 1
    δ[2, 2, 2, 2] = 1 

    @tensor T[-1 -2 ; -3 -4] := sqrt_t[-1; 1] * sqrt_t[-2; 2] * sqrt_t[3; -3] * sqrt_t[4; -4] * δ[1 2; 3 4]
    return T
end

T = tensor_square_ising(asinh(1) / 2)
A = TensorMap(rand, ComplexF64, ℂ^4*ℂ^2, ℂ^4) 

AL, AR = vumps(A, T; ad_steps=20)
AC, C = vumps_update(AL, AR, T)
AL1, AR1, _ = mps_update(AC, C)
using MPSKit
ψl1 = InfiniteMPS([AL1])
ψl = InfiniteMPS([AL])
ψr1 = InfiniteMPS([AR1])
ψr = InfiniteMPS([AR])
@show norm(dot(ψl1, ψl))
@show norm(dot(ψr1, ψr))
@show norm(dot(ψl, ψr))

AL2 = gauge_fixing_L(AL, AL1)
AL.data ./ AL2.data 

AL - AL2 |> norm
AR2 = gauge_fixing_R(AR, AR1)
AR - AR2 |> norm

function vumps1(A, T; maxiter=500, ad_steps=10, tol=1e-12)
    # TODO.: canonical form conversion
    AL, AR = ignore_derivatives() do
        sp = domain(A)[1]
        C = TensorMap(rand, ComplexF64, sp, sp)
        AL, AR = mps_update(A, C)

        conv_meas = 999
        ix = 0
        while conv_meas > tol && ix < maxiter
            ix += 1
            AC, C = vumps_update(AL, AR, T)
            AL, AR, conv_meas = mps_update(AC, C)
            println(ix, ' ', conv_meas)
        end
        return AL, AR
    end
    
    for _ in 1:ad_steps
        AC, C = vumps_update(AL, AR, T)
        AL1, AR1, _ = mps_update(AC, C)
        AL = gauge_fixing_L(AL, AL1)
        AR = gauge_fixing_R(AR, AR1)
        @tensor vl[-1] := AL[1 -1; 1]
        ψl1 = InfiniteMPS([AL1])
        ψl = InfiniteMPS([AL])
        ψr1 = InfiniteMPS([AR1])
        ψr = InfiniteMPS([AR])
        @show norm(dot(ψl1, ψl))
        @show norm(dot(ψr1, ψr))
        @show norm(dot(ψl, ψr))
        @show norm(vl)
    end
    return AL, AR
end

function _F(T)
    AL, AR = vumps(A, T; ad_steps=100)
    @tensor vl[-1] := AL[1 -1; 1]
    return norm(vl)
end
    
_F(T)
_F(T)
_F(T)
ad1 = _F'(T)
ad2 = _F'(T)
ad3 = _F'(T)

ad1.data .- ad2.data

norm(ad1)
norm(ad2)
norm(ad3)
ad1 - ad3 |> norm
ad1 - ad2 |> norm
ad2 - ad3 |> norm
