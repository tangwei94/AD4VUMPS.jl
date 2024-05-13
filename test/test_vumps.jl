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

    βc = asinh(1) / 2
    T = tensor_square_ising(βc)
    
    function _F1(AL1)
        AC, C = vumps_update(AL1, AR, T)
        return norm(tr(QAC' * AC)) / norm(AC) #+ norm(tr(C)) / norm(C)
    end
    function _F2(AR1)
        AC, C = vumps_update(AL, AR1, T)
        return norm(tr(QAC' * AC)) / norm(AC) #+ norm(tr(C)) / norm(C)
    end
    function _F3(T1)
        AC, C = vumps_update(AL, AR, T1)
        return norm(tr(QAC' * AC)) / norm(AC) #+ norm(tr(C)) / norm(C)
    end
  
    test_ADgrad(_F1, AL)
    test_ADgrad(_F2, AR)
    test_ADgrad(_F3, T)
end