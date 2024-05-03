abstract type AbstractTransferMatrix end

struct MPSMPS_TransferMatrix{A<:AbstractTensorMap,C<:AbstractTensorMap} <:
       AbstractTransferMatrix
    above::A
    below::C
    isflipped::Bool
end
struct MPSMPOMPS_TransferMatrix{A<:AbstractTensorMap,B<:AbstractTensorMap,C<:AbstractTensorMap} <:
       AbstractTransferMatrix
    above::A
    middle::B
    below::C
    isflipped::Bool
end
#struct MPSMPS_TransferMatrixBackward

function TensorKit.flip(TM::MPSMPS_TransferMatrix)
    return MPSMPS_TransferMatrix(TM.above, TM.below, true)
end
function TensorKit.flip(TM::MPSMPOMPS_TransferMatrix)
    return MPSMPOMPS_TransferMatrix(TM.above, TM.middle, TM.below, true)
end

function (TM::MPSMPOMPS_TransferMatrix)(v)
    if TM.isflipped == false # right eigenvector
        @tensor Tv[-1 -2; -3] := TM.below[-1 3 ; 1] * TM.middle[-2 5; 3 2] * conj(TM.above[-3 5; 4]) * v[1 2; 4]
        return Tv
    else # left eigenvector
        @tensor Tv[-1; -2 -3] := TM.below[4 5; -3] * TM.middle[2 3; 5 -2] * conj(TM.above[1 3; -1]) * v[1; 2 4]
        return Tv
    end
end

function (TM::MPSMPS_TransferMatrix)(v)
    if TM.isflipped == false # right eigenvector
        @tensor Tv[-1; -2] := TM.below[-1 3; 1] * conj(TM.above[-2 3; 2]) * v[1; 2]
        return Tv
    else # left eigenvector
        @tensor Tv[-1; -2] := TM.below[2 3; -2] * conj(TM.above[1 3; -1]) * v[1; 2]
        return Tv
    end
end

function right_env(TM::MPSMPS_TransferMatrix)
    space_above = domain(TM.above)[1]
    space_below = domain(TM.below)[1]

    init = TensorMap(rand, ComplexF64, space_below, space_above)
    _, ρrs, _ = eigsolve(TM, init, 1, :LM)

    return ρrs[1]
end

function left_env(TM::MPSMPS_TransferMatrix)
    space_above = domain(TM.above)[1]
    space_below = domain(TM.below)[1]

    init = TensorMap(rand, ComplexF64, space_above, space_below)
    _, ρls, _ = eigsolve(flip(TM), init, 1, :LM)

    return ρls[1]
end

function right_env(TM::MPSMPOMPS_TransferMatrix)
    space_above = domain(TM.above)[1]
    space_below = domain(TM.below)[1]
    space_middle = domain(TM.middle)[1]

    init = TensorMap(rand, ComplexF64, space_below*space_middle, space_above)
    _, ρrs, _ = eigsolve(TM, init, 1, :LM)

    return ρrs[1]
end

function left_env(TM::MPSMPOMPS_TransferMatrix)
    space_above = domain(TM.above)[1]
    space_below = domain(TM.below)[1]
    space_middle = domain(TM.middle)[1]

    init = TensorMap(rand, ComplexF64, space_above, space_middle*space_below)
    _, ρls, _ = eigsolve(flip(TM), init, 1, :LM)

    return ρls[1]
end











