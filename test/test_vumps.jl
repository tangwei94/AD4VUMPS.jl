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
        return norm(tr(QAC' * AC)) / norm(AC) + norm(tr(C)) / norm(C)
    end
    function _F2(AR1)
        AC, C = vumps_update(AL, AR1, T)
        return norm(tr(QAC' * AC)) / norm(AC) + norm(tr(C)) / norm(C)
    end
    function _F3(T1)
        AC, C = vumps_update(AL, AR, T1)
        return norm(tr(QAC' * AC)) / norm(AC) + norm(tr(C)) / norm(C)
    end
  
    test_ADgrad(_F1, AL)
    test_ADgrad(_F2, AR)
    test_ADgrad(_F3, T)
end

@testset "test AD for one vumps step" for ix in 1:10

    sp1 = ℂ^4;
    sp2 = ℂ^2;

    AC0 = TensorMap(rand, ComplexF64, sp1*sp2, sp1);
    C0 = TensorMap(rand, ComplexF64, sp1, sp1);
    AL, AR, _ = mps_update(AC0, C0);

    function _F(T; AL1=AL, AR1=AR)
        AC1, C1 = vumps_update(AL1, AR1, T)
        AL2, AR2, _ = mps_update(AC1, C1)

        @tensor vl[-1] := AL2[1 -1; 1]
        return norm(vl) / norm(AL2)
    end

    T = tensor_square_ising(asinh(1) / 2)
    test_ADgrad(_F, T; tol=1e-7)
end

@testset "test vumps forward" for ix in 1:10
    s = random_real_symmetric_tensor(2) 
    T = tensor_square_ising(asinh(1) / 2) 
    if ix > 1
        T = T + rand() * 0.1 * s
    end
    A = TensorMap(rand, ComplexF64, ℂ^6*ℂ^2, ℂ^6) 
    AL, AR = vumps(T; A=A, verbosity=0)
    ϕ = InfiniteMPS([AL])

    AC, C = vumps_update(AL, AR, T)
    AL, AR, conv_meas = mps_update(AC, C)
    @test conv_meas < 1e-12

    ψi = InfiniteMPS([A])
    ψ, _ = leading_boundary(ψi, DenseMPO([T]), VUMPS(verbosity=0))

    @test log(norm(dot(ψ, ϕ))) < 1e-12
end

@testset "test gauge fixed vumps iteration" for ix in 1:10
    T = tensor_square_ising(asinh(1) / 2)
    A = TensorMap(rand, ComplexF64, ℂ^4*ℂ^2, ℂ^4) 

    O = tensor_square_ising_O(asinh(1) / 2 / 2)
    AL, AR = vumps(T; A=A, verbosity=0)
    
    function _F2(T)
        AL1, AR1 = AD4VUMPS.gauge_fixed_vumps_iteration(AL, AR, T)
        TM = MPSMPOMPSTransferMatrix(AL1, T, AL1)
        EL = left_env(TM)
        ER = right_env(TM)

        @tensor a = EL[4; 1 2] * AL1[1 3; 6] * O[2 5; 3 8] * conj(AL1[4 5; 7]) * ER[6 8; 7]
        @tensor b = EL[4; 1 2] * AL1[1 3; 6] * T[2 5; 3 8] * conj(AL1[4 5; 7]) * ER[6 8; 7]
        return real(a/b)
    end
   
    sX = random_real_symmetric_tensor(2)
    test_ADgrad(_F2, T; sX=sX, num=2, α=1e-4, tol=1e-4)
    test_ADgrad(_F2, T; sX=sX, num=2, α=1e-5, tol=1e-5)
end

@testset "test ad of vumps" for ix in 1:1
    T = tensor_square_ising(asinh(1) / 2)
    A = TensorMap(rand, ComplexF64, ℂ^4*ℂ^2, ℂ^4) 

    O = tensor_square_ising_O(asinh(1) / 2 / 2)
    
    function _F1(T)
        AL1, AR1 = vumps(T; A=A, verbosity=0)
        TM = MPSMPOMPSTransferMatrix(AL1, T, AL1)
        EL = left_env(TM)
        ER = right_env(TM)

        @tensor a = EL[4; 1 2] * AL1[1 3; 6] * O[2 5; 3 8] * conj(AL1[4 5; 7]) * ER[6 8; 7]
        @tensor b = EL[4; 1 2] * AL1[1 3; 6] * T[2 5; 3 8] * conj(AL1[4 5; 7]) * ER[6 8; 7]
        return real(a/b)
    end
   
    sX = random_real_symmetric_tensor(2)
    test_ADgrad(_F1, T; sX=sX, num=2, α=1e-4, tol=1e-4)
    test_ADgrad(_F1, T; sX=sX, num=2, α=1e-5, tol=1e-5)
end