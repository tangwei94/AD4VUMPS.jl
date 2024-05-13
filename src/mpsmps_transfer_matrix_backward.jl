struct MPSMPSTransferMatrixBackward
    VLs::Vector{<:RhoTensor}
    VRs::Vector{<:RhoTensor}
end

Base.:+(bTM1::MPSMPSTransferMatrixBackward, bTM2::MPSMPSTransferMatrixBackward) = MPSMPSTransferMatrixBackward([bTM1.VLs; bTM2.VLs], [bTM1.VRs; bTM2.VRs])
Base.:-(bTM1::MPSMPSTransferMatrixBackward, bTM2::MPSMPSTransferMatrixBackward) = MPSMPSTransferMatrixBackward([bTM1.VLs; bTM2.VLs], [bTM1.VRs; -1*bTM2.VRs])
Base.:*(a::Number, bTM::MPSMPSTransferMatrixBackward) = MPSMPSTransferMatrixBackward(bTM.VLs, a * bTM.VRs)
Base.:*(bTM::MPSMPSTransferMatrixBackward, a::Number) = MPSMPSTransferMatrixBackward(bTM.VLs, a * bTM.VRs)

function right_env_backward(TM::MPSMPSTransferMatrix, λ::Number, vr::RhoTensor, ∂vr::RhoTensor)
    init = similar(vr)
    randomize!(init); 
    init = init - dot(vr, init) * vr # important. the subtracted part lives in the null space of flip(TM) - λ*I
    
    (norm(dot(vr, ∂vr)) > 1e-9) && @warn "right_env_backward: forward computation not gauge invariant: final computation should not depend on the phase of vr." # important
    #∂vr = ∂vr - dot(vr, ∂vr) * vr 
    ξr, info = linsolve(x -> flip(TM)(x) - λ*x, ∂vr', init') # ξr should live in the space of vl
    (info.converged == 0) && @warn "right_env_backward not converged: normres = $(info.normres)"
    
    return ξr
end

function left_env_backward(TM::MPSMPSTransferMatrix, λ::Number, vl::RhoTensor, ∂vl::RhoTensor)
    init = similar(vl)
    randomize!(init); 
    init = init - dot(vl, init) * vl # important

    (norm(dot(vl, ∂vl)) > 1e-9) && @warn "left_env_backward: forward computation not gauge invariant: final computation should not depend on the phase of vl." # important
    ξl, info = linsolve(x -> TM(x) - λ*x, ∂vl', init') # ξl should live in the space of vr
    (info.converged == 0) && @warn "left_env_backward not converged: normres = $(info.normres)"

    return ξl
end

function ChainRulesCore.rrule(::typeof(right_env), TM::MPSMPSTransferMatrix)
    space_above = domain(TM.above)[1]
    space_below = domain(TM.below)[1]

    init = TensorMap(rand, ComplexF64, space_below, space_above)
    λrs, vrs, _ = eigsolve(TM, init, 1, :LM)
    λr, vr = λrs[1], vrs[1]

    function right_env_pushback(∂vr)
        ξr = right_env_backward(TM, λr, vr, ∂vr)
        return NoTangent(), MPSMPSTransferMatrixBackward([-ξr], [vr])
    end
    return vr, right_env_pushback
end

function ChainRulesCore.rrule(::typeof(left_env), TM::MPSMPSTransferMatrix)

    space_above = domain(TM.above)[1]
    space_below = domain(TM.below)[1]

    init = TensorMap(rand, ComplexF64, space_above, space_below)
    λls, vls, _ = eigsolve(flip(TM), init, 1, :LR)
    λl, vl = λls[1], vls[1]
   
    function left_env_pushback(∂vl)
        ξl = left_env_backward(TM, λl, vl, ∂vl)
        return NoTangent(), MPSMPSTransferMatrixBackward([vl], [-ξl])
    end
    return vl, left_env_pushback
end

function ChainRulesCore.rrule(::Type{MPSMPSTransferMatrix}, Au::MPSTensor, Ad::MPSTensor, isflipped::Bool)

    TM = MPSMPSTransferMatrix(Au, Ad, false) # TODO. I should remove isflipped from the struct
    
    function TransferMatrix_pushback(∂TM)
        ∂Au = zero(Au)
        ∂Ad = zero(Ad)
        for (VL, VR) in zip(∂TM.VLs, ∂TM.VRs)
            @tensor ∂Ad_j[-1 -2; -3] := VL'[-1; 1] * Au[1 -2; 2] * VR'[2; -3]
            @tensor ∂Au_j[-1 -2; -3] := VL[-1; 1] * Ad[1 -2; 2] * VR[2; -3]
            ∂Au += ∂Au_j
            ∂Ad += ∂Ad_j
        end
        return NoTangent(), ∂Au, ∂Ad, NoTangent()
    end
    return TM, TransferMatrix_pushback 
end
