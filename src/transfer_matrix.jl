abstract type AbstractTransferMatrix end

struct MPSMPSTransferMatrix{A<:MPSTensor,C<:MPSTensor} <:
       AbstractTransferMatrix
    above::A
    below::C
    isflipped::Bool
end
struct MPSMPOMPSTransferMatrix{A<:MPSTensor,B<:MPOTensor,C<:MPSTensor} <:
       AbstractTransferMatrix
    above::A
    middle::B
    below::C
    isflipped::Bool
end
#struct MPSMPSTransferMatrixBackward

function TensorKit.flip(TM::MPSMPSTransferMatrix)
    return MPSMPSTransferMatrix(TM.above, TM.below, true)
end
function TensorKit.flip(TM::MPSMPOMPSTransferMatrix)
    return MPSMPOMPSTransferMatrix(TM.above, TM.middle, TM.below, true)
end

function (TM::MPSMPSTransferMatrix)(v)
    if TM.isflipped == false # right eigenvector
        @tensor Tv[-1; -2] := TM.below[-1 3; 1] * conj(TM.above[-2 3; 2]) * v[1; 2]
        return Tv
    else # left eigenvector
        @tensor Tv[-1; -2] := TM.below[2 3; -2] * conj(TM.above[1 3; -1]) * v[1; 2]
        return Tv
    end
end

function (TM::MPSMPOMPSTransferMatrix)(v)
    if TM.isflipped == false # right eigenvector
        @tensor Tv[-1 -2; -3] := TM.below[-1 3 ; 1] * TM.middle[-2 5; 3 2] * conj(TM.above[-3 5; 4]) * v[1 2; 4]
        return Tv
    else # left eigenvector
        @tensor Tv[-1; -2 -3] := TM.below[4 5; -2] * TM.middle[2 3; 5 -3] * conj(TM.above[1 3; -1]) * v[1; 4 2]
        return Tv
    end
end

function right_env(TM::MPSMPSTransferMatrix)
    space_above = domain(TM.above)[1]
    space_below = domain(TM.below)[1]

    init = TensorMap(rand, ComplexF64, space_below, space_above)
    _, ρrs, _ = eigsolve(TM, init, 1, :LM)

    return ρrs[1]
end

function left_env(TM::MPSMPSTransferMatrix)
    space_above = domain(TM.above)[1]
    space_below = domain(TM.below)[1]

    init = TensorMap(rand, ComplexF64, space_above, space_below)
    _, ρls, _ = eigsolve(flip(TM), init, 1, :LM)

    return ρls[1]
end

function right_env(TM::MPSMPOMPSTransferMatrix)
    space_above = domain(TM.above)[1]
    space_below = domain(TM.below)[1]
    space_middle = domain(TM.middle)[1]

    init = TensorMap(rand, ComplexF64, space_below*space_middle, space_above)
    _, ρrs, _ = eigsolve(TM, init, 1, :LM)

    return ρrs[1]
end

function left_env(TM::MPSMPOMPSTransferMatrix)
    space_above = domain(TM.above)[1]
    space_below = domain(TM.below)[1]
    space_middle = domain(TM.middle)[1]

    init = TensorMap(rand, ComplexF64, space_above, space_below*space_middle)
    _, ρls, _ = eigsolve(flip(TM), init, 1, :LM)

    return ρls[1]
end











