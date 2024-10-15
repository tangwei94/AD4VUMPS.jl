function gauge_fixing(AL1, AL2)
    TM = MPSMPSTransferMatrix(AL1, AL2)
    σ = left_env(TM)
    U, _ = leftorth(σ; alg=QRpos())
    return U
end
#@non_differentiable gauge_fixing(args...)

function overall_u1_phase(T1::AbstractTensorMap, T2::AbstractTensorMap)
    for (f1, f2) in fusiontrees(T2)
        for ix in 1:length(T2[f1, f2])
            if norm(T2[f1, f2][ix]) > 1e-2
                return T1[f1, f2][ix] / T2[f1, f2][ix]
            end
        end
    end
end
function ChainRulesCore.rrule(::typeof(overall_u1_phase), T1::AbstractTensorMap, T2::AbstractTensorMap)
    info = []
    for (f1, f2) in fusiontrees(T2)
        for ix in 1:length(T2[f1, f2])
            if norm(T2[f1, f2][ix]) > 1e-2
                fusiontrees = (f1, f2)
                index = ix
                α = T1[f1, f2][ix] / T2[f1, f2][ix]
                push!(info, (fusiontrees, index, α))
                break
            end
        end
    end
    fusiontrees, index, α = info[1] 

    function overall_u1_phase_pushback(∂α)
        ∂T1 = zero(T1)
        ∂T2 = zero(T2)
        ∂T1[fusiontrees][index] = ∂α / T2[fusiontrees][index]
        ∂T2[fusiontrees][index] = -∂α * T1[fusiontrees][index] / T2[fusiontrees][index]^2
        return NoTangent(), ∂T1, ∂T2
    end

    return α, overall_u1_phase_pushback
end
#@non_differentiable overall_u1_phase(args...)

#function diagonalize_C(AL::MPSTensor, AR::MPSTensor, AC::MPSTensor, C::MPSTensor)
#    U, C1, V = tsvd(C)
#    @tensor AL1[-1 -2; -3] := AL[1 -2; 2] * U'[-1; 1] * V'[2; -3]
#    @tensor AR1[-1 -2; -3] := AR[1 -2; 2] * U'[-1; 1] * V'[2; -3]
#    @tensor AC1[-1 -2; -3] := AC[1 -2; 2] * U'[-1; 1] * V'[2; -3]
#    return AL1, AR1, AC1, C1
#end

function gauge_fixed_vumps_iteration(AL::MPSTensor, AR::MPSTensor, T::MPOTensor)
    AC1, C1 = vumps_update(AL, AR, T)
    AL1, AR1, _ = mps_update(AC1, C1)

    U = gauge_fixing(AL, AL1)
    @tensor AR1_gauged[-1 -2; -3] := AR1[1 -2; 2] * U[-1; 1] * U'[2; -3]
    @tensor AL1_gauged[-1 -2; -3] := AL1[1 -2; 2] * U[-1; 1] * U'[2; -3]

    λ = overall_u1_phase(AL, AL1_gauged)
    AL1_gauged = AL1_gauged * λ
    AR1_gauged = AR1_gauged * λ

    return AL1_gauged, AR1_gauged 
end