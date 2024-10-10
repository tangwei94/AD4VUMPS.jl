abstract type AbstractLinearMap end
#abstract type AbstractLinearMapBackward end

struct LinearMapBackward
    VLs::Vector{<:AbstractTensorMap}
    VRs::Vector{<:AbstractTensorMap}
end

Base.:+(bTM1::LinearMapBackward, bTM2::LinearMapBackward) = LinearMapBackward([bTM1.VLs; bTM2.VLs], [bTM1.VRs; bTM2.VRs])
Base.:-(bTM1::LinearMapBackward, bTM2::LinearMapBackward) = LinearMapBackward([bTM1.VLs; bTM2.VLs], [bTM1.VRs; -1*bTM2.VRs])
Base.:*(a::Number, bTM::LinearMapBackward) = LinearMapBackward(bTM.VLs, a * bTM.VRs)
Base.:*(bTM::LinearMapBackward, a::Number) = LinearMapBackward(bTM.VLs, a * bTM.VRs)

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
    
    (norm(dot(vr, ∂vr)) > 1e-6) && @warn "right_env_backward: forward computation not gauge invariant: final computation should not depend on the phase of vr. err=$(norm(dot(vr, ∂vr)))" 
    ∂vr = ∂vr - dot(vr, ∂vr) * vr 
    ξr, info = linsolve(x -> left_transfer(TM, x) - λ*x, ∂vr', init') # ξr should live in the space of vl
    (info.converged == 0) && @warn "right_env_backward not converged: normres = $(info.normres)"
    
    return ξr
end

function left_env_backward(TM::AbstractLinearMap, λ::Number, vl::AbstractTensorMap, ∂vl::AbstractTensorMap)
    init = similar(vl)
    randomize!(init); 
    init = init - dot(vl, init) * vl # the subtracted part lives in the null space of TM - λ*I

    (norm(dot(vl, ∂vl)) > 1e-6) && @warn "left_env_backward: forward computation not gauge invariant: final computation should not depend on the phase of vl. err=$(norm(dot(vl, ∂vl)))" 
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
