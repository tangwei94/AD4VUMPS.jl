@testset "test gauge_fixing" for _ in 1:10
    A = TensorMap(rand, ComplexF64, ℂ^6*ℂ^2, ℂ^6)
    AL, _ = leftorth(A)
    M = TensorMap(rand, ComplexF64, ℂ^6, ℂ^6)
    U, _ = leftorth(M)

    @tensor AL1[-1 -2; -3] := AL[1 -2; 2] * U'[-1; 1] * U[2; -3]
    
    U1 = gauge_fixing(AL, AL1)
    λ = overall_u1_phase(U, U1)

    @test norm(U - U1 * λ) < 1e-9
end 

@testset "test gauge_fixed_vumps_iteration" for _ in 1:10
    s = random_real_symmetric_tensor(2) 
    T = tensor_square_ising(asinh(1) / 2) 
    A = TensorMap(rand, ComplexF64, ℂ^6*ℂ^2, ℂ^6) 
    AL, AR = vumps(T; A=A, verbosity=0)
    O = tensor_square_ising_O(asinh(1) / 2 / 2)

    AL1, AR1 = AD4VUMPS.gauge_fixed_vumps_iteration(AL, AR, T)
    @test norm(AL1 - AL) < 1e-9
    @test norm(AR1 - AR) < 1e-9

    function _F1(T)
        AL1, AR1 = AD4VUMPS.gauge_fixed_vumps_iteration(AL, AR, T)
        TM = MPSMPOMPSTransferMatrix(AL1, T, AL1)
        EL = left_env(TM)
        ER = right_env(TM)

        @tensor a = EL[4; 1 2] * AL1[1 3; 6] * O[2 5; 3 8] * conj(AL1[4 5; 7]) * ER[6 8; 7]
        @tensor b = EL[4; 1 2] * AL1[1 3; 6] * T[2 5; 3 8] * conj(AL1[4 5; 7]) * ER[6 8; 7]
        return real(a/b)
    end
    
    sX = random_real_symmetric_tensor(2)
    test_ADgrad(_F1, T; sX=sX, num=2, α=1e-4, tol=1e-4)
end