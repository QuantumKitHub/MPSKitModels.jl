using MPSKitModels
using TensorKit
using TensorOperations
using Test

cutoff = 3
elt = ComplexF64

@testset "non-symmetric bosonic operators" begin
    raising = a_plus(; cutoff=cutoff)
    lowering = a_min(; cutoff=cutoff)
    @test raising' ≈ lowering
    @test a_number(; cutoff=cutoff) ≈ raising * lowering atol = 1e-4
end

@testset "U1-symmetric bosonic operators" begin
    @test convert(Array, a_number(U1Irrep; cutoff=cutoff)) ≈ convert(Array, a_number(; cutoff=cutoff))
    @test permute(a_plus(U1Irrep; cutoff=cutoff, side=:L)', (2, 1), (3,)) ≈ a_min(U1Irrep; cutoff=cutoff, side=:R)
    @test permute(a_min(U1Irrep; cutoff=cutoff, side=:L)', (2, 1), (3,)) ≈ a_plus(U1Irrep; cutoff=cutoff, side=:R)
end
