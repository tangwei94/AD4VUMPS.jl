@testset "test backward for right_env" for ix in 1:10
    sp1 = ℂ^12;
    sp2 = ℂ^3;
    A = TensorMap(rand, ComplexF64, sp1*sp2, sp1);
    B = TensorMap(rand, ComplexF64, sp1*sp2, sp1);
    M = TensorMap(rand, ComplexF64, sp2*sp2, sp2*sp2);

    Q = TensorMap(rand, ComplexF64, sp1, sp1*sp2)

    function _F1(X)
        TM1 = MPSMPOMPSTransferMatrix(X, M, X, false)
        v1 = right_env(TM1; init=A)
        return norm(tr(Q * v1)) / norm(v1) 
    end
    function _F2(X)
        TM1 = MPSMPOMPSTransferMatrix(A, M, X, false)
        v1 = right_env(TM1)
        return norm(tr(Q * v1)) / norm(v1) 
    end
    function _F3(X)
        TM1 = MPSMPOMPSTransferMatrix(A, X, B, false)
        v1 = right_env(TM1)
        return norm(tr(Q * v1)) / norm(v1) 
    end

    test_ADgrad(_F1, A)
    test_ADgrad(_F1, B)
    test_ADgrad(_F2, A)
    test_ADgrad(_F2, B)
    test_ADgrad(_F3, M)
end

@testset "test backward for left_env" for ix in 1:10
    sp1 = ℂ^12;
    sp2 = ℂ^3;

    A = TensorMap(rand, ComplexF64, sp1*sp2, sp1);
    B = TensorMap(rand, ComplexF64, sp1*sp2, sp1);
    M = TensorMap(rand, ComplexF64, sp2*sp2, sp2*sp2);

    Q = TensorMap(rand, ComplexF64, sp1*sp2, sp1)

    function _F1(X)
        TM1 = MPSMPOMPSTransferMatrix(X, M, X, false)
        v1 = left_env(TM1)
        return norm(tr(Q * v1)) / norm(v1) 
    end
    function _F2(X)
        TM1 = MPSMPOMPSTransferMatrix(A, M, X, false)
        v1 = left_env(TM1)
        return norm(tr(Q * v1)) / norm(v1) 
    end
    function _F3(X)
        TM1 = MPSMPOMPSTransferMatrix(A, X, B, false)
        v1 = left_env(TM1)
        return norm(tr(Q * v1)) / norm(v1) 
    end

    test_ADgrad(_F1, A)
    test_ADgrad(_F1, B)
    test_ADgrad(_F2, A)
    test_ADgrad(_F2, B)
    test_ADgrad(_F3, M)
end

@testset "test backward for cases where A and B have different virtual bondD" for ix in 1:10
    spa = ℂ^12;
    spb = ℂ^8;
    sp2 = ℂ^3;

    A = TensorMap(rand, ComplexF64, spa*sp2, spa);
    A1 = TensorMap(rand, ComplexF64, spa*sp2, spa);
    B = TensorMap(rand, ComplexF64, spb*sp2, spb);
    B1 = TensorMap(rand, ComplexF64, spb*sp2, spb);
    M = TensorMap(rand, ComplexF64, sp2*sp2, sp2*sp2);

    Qr = TensorMap(rand, ComplexF64, spa, spb*sp2)
    Ql = TensorMap(rand, ComplexF64, spb*sp2, spa)

    function _F1(X)
        TM1 = MPSMPOMPSTransferMatrix(A, X, B, false)
        v1 = left_env(TM1)
        return norm(tr(Ql * v1)) / norm(v1) 
    end
    function _F2(X)
        TM1 = MPSMPOMPSTransferMatrix(A, X, B, false)
        v1 = right_env(TM1)
        return norm(tr(Qr * v1)) / norm(v1) 
    end
    function _Fa1(X)
        TM1 = MPSMPOMPSTransferMatrix(X, M, B, false)
        v1 = right_env(TM1)
        return norm(tr(Qr * v1)) / norm(v1) 
    end
    function _Fa2(X)
        TM1 = MPSMPOMPSTransferMatrix(X, M, B, false)
        v1 = left_env(TM1)
        return norm(tr(Ql * v1)) / norm(v1) 
    end
    function _Fb1(X)
        TM1 = MPSMPOMPSTransferMatrix(A, M, X, false)
        v1 = right_env(TM1)
        return norm(tr(Qr * v1)) / norm(v1) 
    end
    function _Fb2(X)
        TM1 = MPSMPOMPSTransferMatrix(A, M, X, false)
        v1 = left_env(TM1)
        return norm(tr(Ql * v1)) / norm(v1) 
    end

    test_ADgrad(_F1, M)
    test_ADgrad(_F2, M)
    test_ADgrad(_Fa1, A1)
    test_ADgrad(_Fa2, A1)
    test_ADgrad(_Fb1, B1)
    test_ADgrad(_Fb2, B1)
end
