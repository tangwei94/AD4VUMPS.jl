using Test
using TensorKit, TensorOperations, KrylovKit
using ChainRules, ChainRulesCore, Zygote

using Revise
using MPSTransferMatrix

function test_ADgrad(_F, X)

    # retraction direction
    for i in 1:10
        sX = similar(X)
        randomize!(sX)

        # finite diff along retraction direction
        α = 1e-5
        ∂α1 = (_F(X + α * sX) - _F(X - α * sX)) / (2 * α)
        α = 1e-6
        ∂α2 = (_F(X + α * sX) - _F(X - α * sX)) / (2 * α)

        # test correctness of derivative from AD
        ∂X = _F'(X);
        ∂αad = real(dot(∂X, sX))
        @test abs(∂α1 - ∂αad) < 1e-5
        @test abs(∂α2 - ∂αad) < 1e-6
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
