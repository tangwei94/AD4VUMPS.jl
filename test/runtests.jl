using Test
using TensorKit, TensorOperations, KrylovKit
using ChainRules, ChainRulesCore, Zygote

using Revise
using AD4VUMPS

function test_ADgrad(_F, X; α = 1e-4, tol = 1e-8, sX = nothing, num = 10)

    # retraction direction
    for i in 1:num
        if isnothing(sX)
            sX = similar(X)
            randomize!(sX)
        end
        sX = sX / norm(sX)

        # finite diff along retraction direction
        ∂α1 = (-_F(X + 2*α * sX) + 8*_F(X + α * sX) - 8*_F(X - α * sX) + _F(X - 2*α * sX)) / (12 * α)

        # test correctness of derivative from AD
        ∂X = _F'(X);
        ∂αad = real(dot(∂X, sX))
        @show abs(∂α1 - ∂αad)
        @test abs(∂α1 - ∂αad) < tol 
        if !(abs(∂α1 - ∂αad) < tol)
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
function tensor_square_ising_O(β::Real)
    t = TensorMap(ComplexF64[exp(β) exp(-β); exp(-β) exp(β)], ℂ^2, ℂ^2)
    sqrt_t = sqrt(t)
    δ = TensorMap(zeros, ComplexF64, ℂ^2*ℂ^2, ℂ^2*ℂ^2)

    δ.data .= 1

    @tensor T[-1 -2 ; -3 -4] := sqrt_t[-1; 1] * sqrt_t[-2; 2] * sqrt_t[3; -3] * sqrt_t[4; -4] * δ[1 2; 3 4]
    return T
end

function random_real_symmetric_tensor(d::Int)
    O_dat = rand(Float64, d, d, d, d)
    #O_dat = O_dat + permutedims(O_dat, (2, 1, 4, 3)) + permutedims(O_dat, (3, 4, 1, 2)) + permutedims(O_dat, (4, 3, 2, 1))
    O_dat = O_dat + permutedims(O_dat, (1, 3, 2, 4))

    O = TensorMap(O_dat, ℂ^d*ℂ^d, ℂ^d*ℂ^d)
    return O / norm(O)
end

include("test_mpsmps_transfer_matrix.jl");
include("test_mpsmpomps_transfer_matrix.jl");
include("test_ACMap.jl");
include("test_vumps.jl");

@testset "Aqua" verbose = true begin
    using Aqua
    Aqua.test_all(
    AD4VUMPS;
    ambiguities=false,
    stale_deps=false, # FIXME. disable stale_deps for now. 
    deps_compat=false, # FIXME. disable deps_compat for now.
  )
end
