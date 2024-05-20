struct ACMap{TypeL<:EnvTensorL, TypeT<:MPOTensor, TypeR<:EnvTensorR} <:AbstractLinearMap
    EL::TypeL
    T::TypeT
    ER::TypeR
end

function left_space(TM::ACMap)
    space_vL = domain(TM.EL)[1]
    space_ph = domain(TM.T)[2]
    space_vR = domain(TM.ER)[1]

    return space_vL*space_ph←space_vR
end
function right_space(TM::ACMap)
    return adjoint(left_space(TM))
end
function left_transfer(TM::ACMap, v::AbstractTensorMap)
    @tensor Tv[-1 -2; -3] := TM.EL[-1; 1 4] * TM.T[4 -2; 3 5] * TM.ER[2 5; -3] * v[1 3; 2]
    return Tv
end
function right_transfer(TM::ACMap, v::AbstractTensorMap)
    @tensor Tv[-1; -2 -3] := TM.EL[2; -2 5] * TM.T[5 3; -3 4] * TM.ER[-1 4; 1] * v[1; 2 3]
    return Tv
end
function ChainRulesCore.rrule(::Type{ACMap}, EL::EnvTensorL, T::MPOTensor, ER::EnvTensorR)
    TM = ACMap(EL, T, ER)

    function ACMap_pushback(∂TM)
        ∂EL = zero(EL)
        ∂T = zero(T)
        ∂ER = zero(ER)
        for (VL, VR) in zip(∂TM.VLs, ∂TM.VRs)
            @tensor ∂EL_j[-1; -2 -3] := VL'[4; -2 5] * conj(T[-3 2; 5 3]) * VR'[-1 2; 1] * conj(ER[4 3; 1])
            @tensor ∂T_j[-1 -2; -3 -4] := conj(EL[4; 1 -1]) * VL'[2; 1 -3] * conj(ER[2 -4; 3]) * VR'[4 -2; 3]
            @tensor ∂ER_j[-1 -2; -3] := conj(EL[4; 1 3]) * VR'[4 5; -3] * conj(T[3 5; 2 -2]) * VL'[-1; 1 2]
            ∂EL += ∂EL_j
            ∂T += ∂T_j
            ∂ER += ∂ER_j
        end
        return NoTangent(), ∂EL, ∂T, ∂ER
    end
    return TM, ACMap_pushback 
end

fixed_point(TM::ACMap) = left_env(TM)

struct CMap{TypeL<:EnvTensorL, TypeR<:EnvTensorR}
    EL::TypeL
    ER::TypeR
end
