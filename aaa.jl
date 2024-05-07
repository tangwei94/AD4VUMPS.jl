using TensorKit, TensorOperations, KrylovKit
using ChainRules, ChainRulesCore, Zygote
using Revise
using MPSTransferMatrix

sp1 = ℂ^6;
sp2 = ℂ^2;

A = TensorMap(rand, ComplexF64, sp1*sp2, sp1);
M = TensorMap(rand, ComplexF64, sp2*sp2, sp2*sp2);

tm = MPSMPSTransferMatrix(A, A, false);
v = TensorMap(rand, ComplexF64, sp1, sp1);

tm(v)
right_env(tm)
left_env(tm)

function aaa(A)
    tm = MPSMPSTransferMatrix(A, A, false);
    vr = right_env(tm)
    return norm(tr(vr)) / norm(vr)
end

function aaa2(TM)
    vr = right_env(TM)
    return norm(tr(vr)) / norm(vr)
end

aaa(A)
aaa'(A)

