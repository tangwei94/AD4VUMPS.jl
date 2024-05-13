@testset "test AD for mps_update" for ix in 1:10
    sp1 = ℂ^4;
    sp2 = ℂ^2;

    AC = TensorMap(rand, ComplexF64, sp1*sp2, sp1);
    C = TensorMap(rand, ComplexF64, sp1, sp1);
    QAC = TensorMap(rand, ComplexF64, sp1*sp2, sp1);

    function _F1(C1)
        AL, AR, _ = mps_update(AC, C1)
        return norm(tr(QAC' * AL)) + norm(tr(QAC * AR'))
    end
    function _F2(AC1)
        AL, AR, _ = mps_update(AC1, C)
        return norm(tr(QAC' * AL)) + norm(tr(QAC * AR'))
    end
  
    test_ADgrad(_F1, C)
    test_ADgrad(_F2, AC)
end

@testset "test AD for vumps_update" for ix in 1:10
    sp1 = ℂ^4;
    sp2 = ℂ^2;

    AC0 = TensorMap(rand, ComplexF64, sp1*sp2, sp1);
    C0 = TensorMap(rand, ComplexF64, sp1, sp1);
    AL, AR, _ = mps_update(AC0, C0);
    QAC = TensorMap(rand, ComplexF64, sp1*sp2, sp1);
    QC = TensorMap(rand, ComplexF64, sp1, sp1);

    function tensor_square_ising(β::Real)
        t = TensorMap(ComplexF64[exp(β) exp(-β); exp(-β) exp(β)], ℂ^2, ℂ^2)
        sqrt_t = sqrt(t)
        δ = TensorMap(zeros, ComplexF64, ℂ^2*ℂ^2, ℂ^2*ℂ^2)

        δ[1, 1, 1, 1] = 1
        δ[2, 2, 2, 2] = 1 

        @tensor T[-1 -2 ; -3 -4] := sqrt_t[-1; 1] * sqrt_t[-2; 2] * sqrt_t[3; -3] * sqrt_t[4; -4] * δ[1 2; 3 4]
        return T
    end
    βc = asinh(1) / 2
    T = tensor_square_ising(βc)
        
    function _F1(AL)
        TM_L = MPSMPOMPSTransferMatrix(AL, T, AL, false)
        TM_R = MPSMPOMPSTransferMatrix(AR, T, AR, false)
        EL = left_env(TM_L)
        ER = right_env(TM_R)
        C_map = MPSMPSTransferMatrix(EL', ER, false)
        C = left_env(C_map) 
        #return norm(tr(QAC' * AC)) / norm(AC) #+ norm(tr(C)) / norm(C)
        return norm(tr(QC*C)) / norm(C)
    end
    function _F2(AR1)
        AC, C = vumps_update(AL, AR1, T)
        #return norm(tr(QAC' * AC)) / norm(AC) #+ norm(tr(C)) / norm(C)
        return norm(tr(QC*C)) / norm(C)
    end
    function _F3(T1)
        AC, C = vumps_update(AL, AR, T1)
        #return norm(tr(QAC' * AC)) / norm(AC) #+ norm(tr(C)) / norm(C)
        return norm(tr(QC*C)) / norm(C)
    end
  
    test_ADgrad(_F1, AL)
    #test_ADgrad(_F2, AR)
    #test_ADgrad(_F3, T)
end