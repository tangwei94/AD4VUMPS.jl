struct MPSMPOMPSTransferMatrix{A<:MPSTensor,B<:MPOTensor,C<:MPSTensor} <:
       AbstractLinearMap
    above::A
    middle::B
    below::C
end

function right_space(TM::MPSMPOMPSTransferMatrix)
    space_above = domain(TM.above)[1]
    space_below = domain(TM.below)[1]
    space_middle = domain(TM.middle)[1]
    return space_below*space_middle←space_above
end
function left_space(TM::MPSMPOMPSTransferMatrix)
    space_above = domain(TM.above)[1]
    space_below = domain(TM.below)[1]
    space_middle = domain(TM.middle)[1]
    return space_above←space_below*space_middle
end

function left_transfer(TM::MPSMPOMPSTransferMatrix, v::AbstractTensorMap)
    @tensor Tv[-1; -2 -3] := TM.below[4 5; -2] * TM.middle[2 3; 5 -3] * conj(TM.above[1 3; -1]) * v[1; 4 2]
    return Tv
end
function right_transfer(TM::MPSMPOMPSTransferMatrix, v::AbstractTensorMap)
    @tensor Tv[-1 -2; -3] := TM.below[-1 3 ; 1] * TM.middle[-2 5; 3 2] * conj(TM.above[-3 5; 4]) * v[1 2; 4]
    return Tv
end

function ChainRulesCore.rrule(::Type{MPSMPOMPSTransferMatrix}, Au::MPSTensor, M::MPOTensor, Ad::MPSTensor)

    TM = MPSMPOMPSTransferMatrix(Au, M, Ad)
    
    function TransferMatrix_pushback(∂TM)
        ∂Au = zero(Au)
        ∂M = zero(M)
        ∂Ad = zero(Ad)
        for (VL, VR) in zip(∂TM.VLs, ∂TM.VRs)
            @tensor ∂Ad_j[-1 -2; -3] := VL'[-1 3; 1] * Au[1 4; 2] * conj(M[3 4; -2 5]) * VR'[2; -3 5]
            @tensor ∂M_j[-1 -2; -3 -4] := VL'[1 -1; 4] * conj(Ad[1 -3; 2]) * VR'[3; 2 -4] * Au[4 -2; 3] 
            @tensor ∂Au_j[-1 -2; -3] := VL[-1; 1 3] * M[3 -2; 4 5] * Ad[1 4; 2] * VR[2 5; -3]
            ∂Au += ∂Au_j
            ∂M += ∂M_j
            ∂Ad += ∂Ad_j
        end
        return NoTangent(), ∂Au, ∂M, ∂Ad
    end
    return TM, TransferMatrix_pushback 
end