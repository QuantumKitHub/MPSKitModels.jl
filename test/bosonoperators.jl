using MPSKitModels
using TensorKit
using TensorOperations
using Test

const cutoff = 3

@testset "no symmetry" begin
    a_plusmin = a_plus(cutoff) ⊗ a_min(cutoff)
    a_minplus = a_min(cutoff) ⊗ a_plus(cutoff)
    @test a_plusmin ≈ a_minplus'
end

@testset "particle number conservation" begin
    @tensor a_plusmin[-1 -2; -3 -4] := a_plus(cutoff, ComplexF64, U₁; side=:left)[-1;
                                                                                  -3 1] *
                                       a_min(cutoff, ComplexF64, U₁; side=:right)[1 -2; -4]

    @tensor a_minplus[-1 -2; -3 -4] := a_min(cutoff, ComplexF64, U₁; side=:left)[-1; -3 1] *
                                       a_plus(cutoff, ComplexF64, U₁; side=:right)[1 -2; -4]
    @test a_plusmin ≈ a_minplus'
end
