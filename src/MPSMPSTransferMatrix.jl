struct MPSMPSTransferMatrix{A<:MPSTensor,C<:MPSTensor} <:
       AbstractLinearMap
    above::A
    below::C
end

function right_space(TM::MPSMPSTransferMatrix)
    space_above = domain(TM.above)[1]
    space_below = domain(TM.below)[1]
    return space_below←space_above
end
function left_space(TM::MPSMPSTransferMatrix)
    space_above = domain(TM.above)[1]
    space_below = domain(TM.below)[1]
    return space_above←space_below
end

function left_transfer(TM::MPSMPSTransferMatrix, v::AbstractTensorMap)
    @tensor Tv[-1; -2] := TM.below[2 3; -2] * conj(TM.above[1 3; -1]) * v[1; 2]
    return Tv
end
function right_transfer(TM::MPSMPSTransferMatrix, v::AbstractTensorMap)
    @tensor Tv[-1; -2] := TM.below[-1 3; 1] * conj(TM.above[-2 3; 2]) * v[1; 2]
    return Tv
end

function ChainRulesCore.rrule(::Type{MPSMPSTransferMatrix}, Au::MPSTensor, Ad::MPSTensor)

    TM = MPSMPSTransferMatrix(Au, Ad) 
    
    function TransferMatrix_pushback(_∂TM)
        ∂TM = unthunk(_∂TM)
        ∂Au = zero(Au)
        ∂Ad = zero(Ad)
        for (VL, VR) in zip(∂TM.VLs, ∂TM.VRs)
            @tensor ∂Ad_j[-1 -2; -3] := VL'[-1; 1] * Au[1 -2; 2] * VR'[2; -3]
            @tensor ∂Au_j[-1 -2; -3] := VL[-1; 1] * Ad[1 -2; 2] * VR[2; -3]
            ∂Au += ∂Au_j
            ∂Ad += ∂Ad_j
        end
        return NoTangent(), ∂Au, ∂Ad
    end
    return TM, TransferMatrix_pushback 
end