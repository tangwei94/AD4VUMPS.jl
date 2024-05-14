struct MPSMPOMPSTransferMatrixBackward
    VLs::Vector{<:EnvTensorL}
    VRs::Vector{<:EnvTensorR}
end

Base.:+(bTM1::MPSMPOMPSTransferMatrixBackward, bTM2::MPSMPOMPSTransferMatrixBackward) = MPSMPOMPSTransferMatrixBackward([bTM1.VLs; bTM2.VLs], [bTM1.VRs; bTM2.VRs])
Base.:-(bTM1::MPSMPOMPSTransferMatrixBackward, bTM2::MPSMPOMPSTransferMatrixBackward) = MPSMPOMPSTransferMatrixBackward([bTM1.VLs; bTM2.VLs], [bTM1.VRs; -1*bTM2.VRs])
Base.:*(a::Number, bTM::MPSMPOMPSTransferMatrixBackward) = MPSMPOMPSTransferMatrixBackward(bTM.VLs, a * bTM.VRs)
Base.:*(bTM::MPSMPOMPSTransferMatrixBackward, a::Number) = MPSMPOMPSTransferMatrixBackward(bTM.VLs, a * bTM.VRs)

function right_env_backward(TM::MPSMPOMPSTransferMatrix, λ::Number, vr::EnvTensorR, ∂vr::EnvTensorR)
    init = similar(vr)
    randomize!(init); 
    init = init - dot(vr, init) * vr # important. the subtracted part lives in the null space of flip(TM) - λ*I
    
    (norm(dot(vr, ∂vr)) > 1e-9) && @warn "right_env_backward: forward computation not gauge invariant: final computation should not depend on the phase of vr." # important
    ∂vr = ∂vr - dot(vr, ∂vr) * vr 
    ξr, info = linsolve(x -> flip(TM)(x) - λ*x, ∂vr', init') # subtle
    (info.converged == 0) && @warn "right_env_backward not converged: normres = $(info.normres)"
    
    return ξr
end

function left_env_backward(TM::MPSMPOMPSTransferMatrix, λ::Number, vl::EnvTensorL, ∂vl::EnvTensorL)
    init = similar(vl)
    randomize!(init); 
    init = init - dot(vl, init) * vl # important

    (norm(dot(vl, ∂vl)) > 1e-9) && @warn "left_env_backward: forward computation not gauge invariant: final computation should not depend on the phase of vl." # important
    ∂vl = ∂vl - dot(vl, ∂vl) * vl 
    ξl, info = linsolve(x -> TM(x) - λ*x, ∂vl', init') # subtle
    (info.converged == 0) && @warn "left_env_backward not converged: normres = $(info.normres)"

    return ξl
end

function ChainRulesCore.rrule(::typeof(right_env), TM::MPSMPOMPSTransferMatrix; init::Union{EnvTensorR, Nothing}=nothing)
    if isnothing(init)
        space_above = domain(TM.above)[1]
        space_below = domain(TM.below)[1]
        space_middle = domain(TM.middle)[1]
        init = TensorMap(rand, ComplexF64, space_below*space_middle, space_above)
    end
    λrs, vrs, _ = eigsolve(TM, init, 1, :LM)
    λr, vr = λrs[1], vrs[1]

    function right_env_pushback(∂vr)
        ξr = right_env_backward(TM, λr, vr, ∂vr)
        return NoTangent(), MPSMPOMPSTransferMatrixBackward([-ξr], [vr])
    end
    return vr, right_env_pushback
end

function ChainRulesCore.rrule(::typeof(left_env), TM::MPSMPOMPSTransferMatrix; init::Union{EnvTensorL, Nothing}=nothing)
    if isnothing(init)
        space_above = domain(TM.above)[1]
        space_below = domain(TM.below)[1]
        space_middle = domain(TM.middle)[1]
        init = TensorMap(rand, ComplexF64, space_above, space_below*space_middle)
    end
    λls, vls, _ = eigsolve(flip(TM), init, 1, :LM)
    λl, vl = λls[1], vls[1]
   
    function left_env_pushback(∂vl)
        ξl = left_env_backward(TM, λl, vl, ∂vl)
        return NoTangent(), MPSMPOMPSTransferMatrixBackward([vl], [-ξl])
    end
    return vl, left_env_pushback
end

function ChainRulesCore.rrule(::Type{MPSMPOMPSTransferMatrix}, Au::MPSTensor, M::MPOTensor, Ad::MPSTensor, isflipped::Bool)

    TM = MPSMPOMPSTransferMatrix(Au, M, Ad, false)
    
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
        return NoTangent(), ∂Au, ∂M, ∂Ad, NoTangent()
    end
    return TM, TransferMatrix_pushback 
end
