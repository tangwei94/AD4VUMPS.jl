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
        AC, C = vumps_update(AL1, AR, T; AC_init=AC0, C_init=C0)
        return norm(tr(QAC' * AC)) / norm(AC) + norm(tr(C)) / norm(C)
    end
    function _F2(AR1)
        AC, C = vumps_update(AL, AR1, T)
        return norm(tr(QAC' * AC)) / norm(AC) + norm(tr(C)) / norm(C)
    end
    function _F3(T1)
        AC, C = vumps_update(AL, AR, T1; AC_init=AC0)
        return norm(tr(QAC' * AC)) / norm(AC) + norm(tr(C)) / norm(C)
    end
  
    test_ADgrad(_F1, AL; α=1e-4, tol=1e-4)
    test_ADgrad(_F2, AR; α=1e-4, tol=1e-4)
    test_ADgrad(_F3, T; α=1e-4, tol=1e-4)
end

@testset "test ad for vumps" for ix in 1:10
    T = tensor_square_ising(asinh(1) / 2)
    A = TensorMap(rand, ComplexF64, ℂ^4*ℂ^2, ℂ^4) 
    AL, AR, AC, C = vumps(A, T)
    
    function _F1(T)
        AL1, AR1, AC1, C1 = vumps_for_ad(T; AL=AL, AR=AR, AC=AC, C=C)
        @tensor vl[-1] := AL1[1 -1; 1]
        return norm(vl) / norm(AL1)
    end
    ad1 = _F1'(T)
    ad2 = _F1'(T)
    @show norm(ad1 - ad2)
    @test norm(ad1 - ad2) < 1e-8
   
    function _F(T)
        AL, AR, AC, C = vumps(A, T)
        AL1, AR1, AC1, C1 = vumps_for_ad(T; AL=AL, AR=AR, AC=AC, C=C)
        @tensor vl[-1] := AL1[1 -1; 1]
        return norm(vl) / norm(AL1)
    end
    ad1 = _F'(T)
    ad2 = _F'(T)
    @test norm(ad1 - ad2) < 1e-8

    for ix in []
        sX_dat = rand(Float64, 2, 2, 2, 2)
        sX_dat = sX_dat + permutedims(sX_dat, (2, 1, 4, 3)) + permutedims(sX_dat, (3, 4, 1, 2)) + permutedims(sX_dat, (4, 3, 2, 1))
        sX_dat = sX_dat + permutedims(sX_dat, (1, 3, 2, 4))
        sX = TensorMap(sX_dat, ℂ^2*ℂ^2, ℂ^2*ℂ^2)
        
        test_ADgrad(_F, T; α=1e-4, tol=1e-4, sX=sX, num=1)
    end
end
