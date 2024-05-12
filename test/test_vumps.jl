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
