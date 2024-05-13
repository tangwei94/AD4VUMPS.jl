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
AL, AR, AC, C = vumps(A, T; ad_steps=10)

function _F(T)
    global AL, AR, AC, C
    AC1, C1 = AC, C
    for _ in 1:10
        AC2, C2 = vumps_update(AL, AR, T)#; AC_init=ignore_derivatives(AC), C_init=ignore_derivatives(C))
        AL, AR, conv_meas = mps_update(AC2, C2)
        AC1, C1 = AC2, C2
    end
    @tensor vl[-1] := AL[1 -1; 1]
    return norm(vl) / norm(AL)
end
    
_F(T)
_F(T)
_F(T)
ad1 = _F'(T)
ad2 = _F'(T)
ad3 = _F'(T)

norm(ad1)
norm(ad2)
norm(ad3)
ad1 - ad3 |> norm
ad1 - ad2 |> norm
ad2 - ad3 |> norm
