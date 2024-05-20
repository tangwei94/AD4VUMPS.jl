@testset "test backward for right_env" for ix in 1:10
    sp1 = ℂ^12;
    sp2 = ℂ^3;
    A = TensorMap(rand, ComplexF64, sp1*sp2, sp1);
    B = TensorMap(rand, ComplexF64, sp1*sp2, sp1);
    vinit = TensorMap(rand, ComplexF64, sp1, sp1);

    function _F1(X)
        TM1 = MPSMPSTransferMatrix(X, X)
        v1 = right_env(TM1)
        return norm(tr(v1)) / norm(v1) 
    end
    function _F2(X)
        TM1 = MPSMPSTransferMatrix(A, X)
        v1 = right_env(TM1)
        return norm(tr(v1)) / norm(v1) 
    end
    function _F3(X)
        TM1 = MPSMPSTransferMatrix(X, A)
        v1 = right_env(TM1)
        return norm(tr(v1)) / norm(v1) 
    end

    test_ADgrad(_F1, A)
    test_ADgrad(_F2, A)
    test_ADgrad(_F3, A)
    test_ADgrad(_F1, B)
    test_ADgrad(_F2, B)
    test_ADgrad(_F3, B)

end

@testset "test backward for left_env" for ix in 1:10
    sp1 = ℂ^12;
    sp2 = ℂ^3;
    A = TensorMap(rand, ComplexF64, sp1*sp2, sp1);
    B = TensorMap(rand, ComplexF64, sp1*sp2, sp1);

    function _F1(X)
        TM1 = MPSMPSTransferMatrix(X, X)
        v1 = left_env(TM1)
        return norm(tr(v1)) / norm(v1) 
    end
    function _F2(X)
        TM1 = MPSMPSTransferMatrix(A, X)
        v1 = left_env(TM1)
        return norm(tr(v1)) / norm(v1) 
    end
    function _F3(X)
        TM1 = MPSMPSTransferMatrix(X, A)
        v1 = left_env(TM1)
        return norm(tr(v1)) / norm(v1) 
    end

    test_ADgrad(_F1, A)
    test_ADgrad(_F2, A)
    test_ADgrad(_F3, A)
    test_ADgrad(_F1, B)
    test_ADgrad(_F2, B)
    test_ADgrad(_F3, B)

end

@testset "test backward for both left_env and right_env" for ix in 1:10
    spa = ℂ^8;
    spb = ℂ^12;
    sp2 = ℂ^3;
    A = TensorMap(rand, ComplexF64, spa*sp2, spa);
    B = TensorMap(rand, ComplexF64, spb*sp2, spb);

    function _F1(X)
        TM1 = MPSMPSTransferMatrix(X, X)
        vl1 = left_env(TM1)
        vr1 = right_env(TM1)
        
        TM2 = MPSMPSTransferMatrix(X, B)
        vl2 = left_env(TM2)
        vr2 = right_env(TM2)
   
        return norm(tr(vl1 * vr1)) / norm(vl1) / norm(vr1) + norm(tr(vl2 * vr2)) / norm(vl2) / norm(vr2) 
    end
    function _F2(X)
        TM1 = MPSMPSTransferMatrix(A, X)
        vl1 = left_env(TM1)
        vr1 = right_env(TM1)
        
        TM2 = MPSMPSTransferMatrix(X, X)
        vl2 = left_env(TM2)
        vr2 = right_env(TM2)
   
        return norm(tr(vl1 * vr1)) / norm(vl1) / norm(vr1) + norm(tr(vl2 * vr2)) / norm(vl2) / norm(vr2)
    end

    test_ADgrad(_F1, A)
    test_ADgrad(_F2, A)
    test_ADgrad(_F1, B)
    test_ADgrad(_F2, B)
end