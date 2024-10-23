@testset "test power_method_alg" for M in [5 10 15 20]
    v0 = [1.0, 1.0, 1.0]
    M = Diagonal([1.0, 0.9, 0.0])
    f(x) = M * x / norm(M * x)
    alg = power_method_alg(M=10, tol=1e-8)
    v1 = iterative_solver(f, v0, alg)
    v_manual = [1.0, 0.9 ^ 10, 0]
    @test norm(v1 - v_manual / norm(v_manual)) < 1e-8
end

@testset "test diis speedup" for λ in [0.5, 0.8, 0.9, 1-1e-2, 1-1e-3, 1-1e-4] 
    v0 = [1.0, 1.0, 1.0]
    M = Diagonal([1.0, λ, 0.0])
    f(x) = M * x / norm(M * x)
    fe(x) = norm(f(x) - x)

    print("power method $(λ)\n")
    alg1 = power_method_alg(M=100, tol=1e-8)
    v1 = iterative_solver(f, v0, alg1)
    err1 = fe(v1)

    print("diis $(λ)\n")
    alg2 = DIIS_extrapolation_alg(; damping_factor = 1e-8)
    v2 = iterative_solver(f, v0, alg2)
    err2 = fe(v2)

    @test err2 < err1 + 1e-8
end
