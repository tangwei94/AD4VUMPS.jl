using LinearAlgebra
using TensorKit, TensorOperations, KrylovKit
using ChainRules, ChainRulesCore, Zygote, OptimKit
using MPSKit
using Revise
using AD4VUMPS

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
  
    withgradient(_F1, T)

    sX = random_real_symmetric_tensor(2)
    test_ADgrad(_F1, T; sX=sX, num=2, α=1e-4, tol=1e-4)
    test_ADgrad(_F1, T; sX=sX, num=2, α=1e-5, tol=1e-5)

    T = tensor_square_ising(asinh(1) / 2)
    A = TensorMap(rand, ComplexF64, ℂ^4*ℂ^2, ℂ^4) 

    O = tensor_square_ising_O(asinh(1) / 2 / 2)
    
    AL, AR = vumps(T; A=A, verbosity=1)

    _, vumps_iteration_vjp = pullback(AD4VUMPS.gauge_fixed_vumps_iteration, AL, AR, T)

    function vjp_ALAR_ALAR(X)
        res = vumps_iteration_vjp((X[1], X[2]))
        return [res[1], res[2]]
    end
    vjp_ALAR_T(X) = vumps_iteration_vjp((X[1], X[2]))[3]
    
    function _F1(AL1, AR1)
        TM = MPSMPOMPSTransferMatrix(AL1, T, AL1)
        EL = left_env(TM)
        ER = right_env(TM)

        @tensor a = EL[4; 1 2] * AL1[1 3; 6] * O[2 5; 3 8] * conj(AL1[4 5; 7]) * ER[6 8; 7]
        @tensor b = EL[4; 1 2] * AL1[1 3; 6] * T[2 5; 3 8] * conj(AL1[4 5; 7]) * ER[6 8; 7]
        return real(a/b)
    end

    _, ∂ALAR = withgradient(_F1, AL, AR)
    ∂AL, ∂AR = ∂ALAR

    X1 = vumps_iteration_vjp(∂ALAR)
        
    function vjp_ALAR_ALAR(X)
        res = vumps_iteration_vjp((X[1], X[2]))
        return [res[1], res[2]]
    end
    vjp_ALAR_T(X) = vumps_iteration_vjp((X[1], X[2]))[3]
    X1 = vjp_ALAR_ALAR([∂AL, ∂AR]) 
    Y1 = [X1[1], X1[2], 1]
    
    VectorInterface.scalartype(a::Vector{Any}) = VectorInterface.scalartype(a[1])
    VectorInterface.inner(a::Vector{Any}, b::Vector{Any}) = sum(VectorInterface.inner.(a, b))
    function f_map(Y)
        Yx = vjp_ALAR_ALAR([Y[1], Y[2]]) + X1
        return [Yx[1], Yx[2], Y[3]]
    end
    vals, vecs, info = eigsolve(f_map, Y1, 3, :LM; tol=1e-6)
    vals
  
    eigsolve(vjp_ALAR_ALAR, X1, 1, :LM; tol=1e-6)    


    f_map1(X) = X - vjp_ALAR_ALAR(X)
    Xsum, info = linsolve(f_map1, X1, X1; tol= 1e-6) # tol cannot be too small
    Y_sol = [Xsum[1], Xsum[2], 1]

    f_map(Y_sol) - Y_sol |> norm