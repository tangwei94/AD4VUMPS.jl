using TensorKit, TensorOperations, KrylovKit
using Revise
using MPSTransferMatrix

sp1 = ℂ^6;
sp2 = ℂ^2;

A = TensorMap(rand, ComplexF64, sp1*sp2, sp1);
M = TensorMap(rand, ComplexF64, sp2*sp2, sp2*sp2);

tm = MPSMPS_TransferMatrix(A, A, false);
v = TensorMap(rand, ComplexF64, sp1, sp1);

tm(v)
right_env(tm)
left_env(tm)

tm = MPSMPOMPS_TransferMatrix(A, M, A, false);
v = TensorMap(rand, ComplexF64, sp1*sp2, sp1);
right_env(tm)
left_env(tm)