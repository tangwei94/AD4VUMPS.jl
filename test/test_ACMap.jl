@testset "test ACMap" for ix in 1:10
    sp1 = ℂ^12;
    sp2 = ℂ^2;

    T = tensor_square_ising(asinh(1) / 2) + 0.01 * random_real_symmetric_tensor(2)
    A = rand(ComplexF64, sp1*sp2, sp1)
    C = rand(ComplexF64, sp1, sp1)
    AL, AR = mps_update(A, C) 

    TM_L = MPSMPOMPSTransferMatrix(AL, T, AL)
    TM_R = MPSMPOMPSTransferMatrix(AR, T, AR)

    EL = left_env(TM_L)
    ER = right_env(TM_R)
    TM1 = ACMap(EL, T, ER)
    left_transfer(TM1, A)
    right_transfer(TM1, A')
end

@testset "test ACMap" for ix in 1:10
    
    sp1 = ℂ^12;
    sp2 = ℂ^2;

    T = tensor_square_ising(asinh(1) / 2) + 0.01 * random_real_symmetric_tensor(2)
    A = rand(ComplexF64, sp1*sp2, sp1)
    C = rand(ComplexF64, sp1, sp1)
    AL, AR = mps_update(A, C) 

    TM_L = MPSMPOMPSTransferMatrix(AL, T, AL)
    TM_R = MPSMPOMPSTransferMatrix(AR, T, AR)

    EL = left_env(TM_L)
    ER = right_env(TM_R)

    function _F1(X)
        TM1 = ACMap(X, T, ER)
        v1 = fixed_point(TM1)
        return norm(tr(A' * v1)) / norm(v1) 
    end
    function _F2(X)
        TM1 = ACMap(EL, X, ER)
        v1 = fixed_point(TM1)
        return norm(tr(A' * v1)) / norm(v1) 
    end
    function _F3(X)
        TM1 = ACMap(EL, T, X)
        v1 = fixed_point(TM1)
        return norm(tr(A' * v1)) / norm(v1) 
    end

    test_ADgrad(_F1, EL)
    test_ADgrad(_F2, T)
    test_ADgrad(_F3, ER)

end