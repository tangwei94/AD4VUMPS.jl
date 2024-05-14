using Test
using TensorKit, TensorOperations, KrylovKit
using ChainRules, ChainRulesCore, Zygote

using Revise
using MPSTransferMatrix

function test_ADgrad(_F, X; α = 1e-4, tol = 1e-8, sX = nothing, num = 10)

    # retraction direction
    for i in 1:num
        if isnothing(sX)
            sX = similar(X)
            randomize!(sX)
        end

        # finite diff along retraction direction
        ∂α1 = (-_F(X + 2*α * sX) + 8*_F(X + α * sX) - 8*_F(X - α * sX) + _F(X - 2*α * sX)) / (12 * α)

        # test correctness of derivative from AD
        ∂X = _F'(X);
        ∂αad = real(dot(∂X, sX))
        @test abs(∂α1 - ∂αad) / abs(∂α1) < tol 
        if !(abs(∂α1 - ∂αad) / abs(∂α1) < tol)
            println("∂α1: ", ∂α1)
            println("∂αad: ", ∂αad)
        end
    end
end

# a function that generates MPO, used in tests
function tensor_square_ising(β::Real)
    t = TensorMap(ComplexF64[exp(β) exp(-β); exp(-β) exp(β)], ℂ^2, ℂ^2)
    sqrt_t = sqrt(t)
    δ = TensorMap(zeros, ComplexF64, ℂ^2*ℂ^2, ℂ^2*ℂ^2)

    δ[1, 1, 1, 1] = 1
    δ[2, 2, 2, 2] = 1 

    @tensor T[-1 -2 ; -3 -4] := sqrt_t[-1; 1] * sqrt_t[-2; 2] * sqrt_t[3; -3] * sqrt_t[4; -4] * δ[1 2; 3 4]
    return T
end

include("test_mpsmps_transfer_matrix.jl");
include("test_mpsmpomps_transfer_matrix.jl");
include("test_vumps.jl");
