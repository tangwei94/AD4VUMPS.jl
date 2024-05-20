abstract type AbstractLinearMap end
#abstract type AbstractLinearMapBackward end

struct MPSMPSTransferMatrix{A<:MPSTensor,C<:MPSTensor} <:
       AbstractLinearMap
    above::A
    below::C
end
struct MPSMPOMPSTransferMatrix{A<:MPSTensor,B<:MPOTensor,C<:MPSTensor} <:
       AbstractLinearMap
    above::A
    middle::B
    below::C
end

struct LinearMapBackward
    VLs::Vector{<:AbstractTensorMap}
    VRs::Vector{<:AbstractTensorMap}
end

Base.:+(bTM1::LinearMapBackward, bTM2::LinearMapBackward) = LinearMapBackward([bTM1.VLs; bTM2.VLs], [bTM1.VRs; bTM2.VRs])
Base.:-(bTM1::LinearMapBackward, bTM2::LinearMapBackward) = LinearMapBackward([bTM1.VLs; bTM2.VLs], [bTM1.VRs; -1*bTM2.VRs])
Base.:*(a::Number, bTM::LinearMapBackward) = LinearMapBackward(bTM.VLs, a * bTM.VRs)
Base.:*(bTM::LinearMapBackward, a::Number) = LinearMapBackward(bTM.VLs, a * bTM.VRs)

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

function left_transfer(TM::MPSMPSTransferMatrix, v::AbstractTensorMap)
    @tensor Tv[-1; -2] := TM.below[2 3; -2] * conj(TM.above[1 3; -1]) * v[1; 2]
    return Tv
end
function right_transfer(TM::MPSMPSTransferMatrix, v::AbstractTensorMap)
    @tensor Tv[-1; -2] := TM.below[-1 3; 1] * conj(TM.above[-2 3; 2]) * v[1; 2]
    return Tv
end
function left_transfer(TM::MPSMPOMPSTransferMatrix, v::AbstractTensorMap)
    @tensor Tv[-1; -2 -3] := TM.below[4 5; -2] * TM.middle[2 3; 5 -3] * conj(TM.above[1 3; -1]) * v[1; 4 2]
    return Tv
end
function right_transfer(TM::MPSMPOMPSTransferMatrix, v::AbstractTensorMap)
    @tensor Tv[-1 -2; -3] := TM.below[-1 3 ; 1] * TM.middle[-2 5; 3 2] * conj(TM.above[-3 5; 4]) * v[1 2; 4]
    return Tv
end

function right_env(TM::AbstractLinearMap)
    init = TensorMap(rand, ComplexF64, right_space(TM))
    _, ρrs, _ = eigsolve(v -> right_transfer(TM, v), init, 1, :LM)
    return ρrs[1]
end
function left_env(TM::AbstractLinearMap)
    init = TensorMap(rand, ComplexF64, left_space(TM))
    _, ρls, _ = eigsolve(v -> left_transfer(TM, v), init, 1, :LM)
    return ρls[1]
end

function right_env_backward(TM::AbstractLinearMap, λ::Number, vr::AbstractTensorMap, ∂vr::AbstractTensorMap)
    init = similar(vr)
    randomize!(init); 
    init = init - dot(vr, init) * vr # the subtracted part lives in the null space of flip(TM) - λ*I
    
    (norm(dot(vr, ∂vr)) > 1e-9) && @warn "right_env_backward: forward computation not gauge invariant: final computation should not depend on the phase of vr." 
    ∂vr = ∂vr - dot(vr, ∂vr) * vr 
    ξr, info = linsolve(x -> left_transfer(TM, x) - λ*x, ∂vr', init') # ξr should live in the space of vl
    (info.converged == 0) && @warn "right_env_backward not converged: normres = $(info.normres)"
    
    return ξr
end

function left_env_backward(TM::AbstractLinearMap, λ::Number, vl::AbstractTensorMap, ∂vl::AbstractTensorMap)
    init = similar(vl)
    randomize!(init); 
    init = init - dot(vl, init) * vl # the subtracted part lives in the null space of TM - λ*I

    (norm(dot(vl, ∂vl)) > 1e-9) && @warn "left_env_backward: forward computation not gauge invariant: final computation should not depend on the phase of vl." 
    ∂vl = ∂vl - dot(vl, ∂vl) * vl 
    ξl, info = linsolve(x -> right_transfer(TM, x) - λ*x, ∂vl', init') # ξl should live in the space of vr
    (info.converged == 0) && @warn "left_env_backward not converged: normres = $(info.normres)"

    return ξl
end

function ChainRulesCore.rrule(::typeof(right_env), TM::AbstractLinearMap)
    init = TensorMap(rand, ComplexF64, right_space(TM))
    λrs, vrs, _ = eigsolve(v -> right_transfer(TM, v), init, 1, :LM)
    λr, vr = λrs[1], vrs[1]

    function right_env_pushback(∂vr)
        ξr = right_env_backward(TM, λr, vr, ∂vr)
        return NoTangent(), LinearMapBackward([-ξr], [vr])
    end
    return vr, right_env_pushback
end

function ChainRulesCore.rrule(::typeof(left_env), TM::AbstractLinearMap)
    init = TensorMap(rand, ComplexF64, left_space(TM))
    λls, vls, _ = eigsolve(v -> left_transfer(TM, v), init, 1, :LM)
    λl, vl = λls[1], vls[1]
   
    function left_env_pushback(∂vl)
        ξl = left_env_backward(TM, λl, vl, ∂vl)
        return NoTangent(), LinearMapBackward([vl], [-ξl])
    end
    return vl, left_env_pushback
end

function ChainRulesCore.rrule(::Type{MPSMPSTransferMatrix}, Au::MPSTensor, Ad::MPSTensor)

    TM = MPSMPSTransferMatrix(Au, Ad) 
    
    function TransferMatrix_pushback(∂TM)
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