function gauge_fixing(AL1, AL2)
    TM = MPSMPSTransferMatrix(AL1, AL2)
    σ = left_env(TM)
    U, _ = leftorth(σ; alg=QRpos())
    return U
end
@non_differentiable gauge_fixing(args...)

function get_u1_phase(T1, T2)
    for (f1, f2) in fusiontrees(T1)
        for ix in 1:length(T1[f1, f2])
            if norm(T1[f1, f2][ix]) > 1e-2
                return angle(T1[f1, f2][ix] / T2[f1, f2][ix])
            end
        end
    end
end
@non_differentiable get_u1_phase(args...)