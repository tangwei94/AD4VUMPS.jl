abstract type AbstractTransferMatrix end

struct MPSMPSTransferMatrix{A<:MPSTensor,C<:MPSTensor} <:
       AbstractTransferMatrix
    above::A
    below::C
end
struct MPSMPOMPSTransferMatrix{A<:MPSTensor,B<:MPOTensor,C<:MPSTensor} <:
       AbstractTransferMatrix
    above::A
    middle::B
    below::C
end

function left_transfer(TM::MPSMPSTransferMatrix, v)
    @tensor Tv[-1; -2] := TM.below[2 3; -2] * conj(TM.above[1 3; -1]) * v[1; 2]
    return Tv
end
function right_transfer(TM::MPSMPSTransferMatrix, v)
    @tensor Tv[-1; -2] := TM.below[-1 3; 1] * conj(TM.above[-2 3; 2]) * v[1; 2]
    return Tv
end
function left_transfer(TM::MPSMPOMPSTransferMatrix, v)
    @tensor Tv[-1; -2 -3] := TM.below[4 5; -2] * TM.middle[2 3; 5 -3] * conj(TM.above[1 3; -1]) * v[1; 4 2]
    return Tv
end
function right_transfer(TM::MPSMPOMPSTransferMatrix, v)
    @tensor Tv[-1 -2; -3] := TM.below[-1 3 ; 1] * TM.middle[-2 5; 3 2] * conj(TM.above[-3 5; 4]) * v[1 2; 4]
    return Tv
end

function right_env(TM::MPSMPSTransferMatrix; init::Union{RhoTensor, Nothing}=nothing)
    if isnothing(init)
        space_above = domain(TM.above)[1]
        space_below = domain(TM.below)[1]
        init = TensorMap(rand, ComplexF64, space_below, space_above)
    end
    _, ρrs, _ = eigsolve(v -> right_transfer(TM, v), init, 1, :LM)

    return ρrs[1]
end

function left_env(TM::MPSMPSTransferMatrix; init::Union{RhoTensor, Nothing}=nothing)
    if isnothing(init)
        space_above = domain(TM.above)[1]
        space_below = domain(TM.below)[1]
        init = TensorMap(rand, ComplexF64, space_above, space_below)
    end
    _, ρls, _ = eigsolve(v -> left_transfer(TM, v), init, 1, :LM)

    return ρls[1]
end

function right_env(TM::MPSMPOMPSTransferMatrix; init::Union{EnvTensorR, Nothing}=nothing)
    if isnothing(init)
        space_above = domain(TM.above)[1]
        space_below = domain(TM.below)[1]
        space_middle = domain(TM.middle)[1]
        init = TensorMap(rand, ComplexF64, space_below*space_middle, space_above)
    end
    _, ρrs, _ = eigsolve(v -> right_transfer(TM, v), init, 1, :LM)

    return ρrs[1]
end

function left_env(TM::MPSMPOMPSTransferMatrix; init::Union{EnvTensorL, Nothing}=nothing)
    if isnothing(init)
        space_above = domain(TM.above)[1]
        space_below = domain(TM.below)[1]
        space_middle = domain(TM.middle)[1]
        init = TensorMap(rand, ComplexF64, space_above, space_below*space_middle)
    end
    _, ρls, _ = eigsolve(v -> left_transfer(TM, v), init, 1, :LM)

    return ρls[1]
end

