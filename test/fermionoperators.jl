using MPSKitModels
using TensorKit
using TensorOperations
using Test
using LinearAlgebra: tr

using MPSKitModels: contract_twosite, contract_onesite
# anticommutation relations
# {cᵢ†, cⱼ†} = 0 = {cᵢ, cⱼ}
# {cᵢ, cⱼ†} = δᵢⱼ

@testset "simple fermions" begin
    cc = contract_twosite(c⁻(; side=:L), c⁻(; side=:R))
    cc⁺ = contract_twosite(c⁻(; side=:L), c⁺(; side=:R))
    c⁺c = contract_twosite(c⁺(; side=:L), c⁻(; side=:R))
    c⁺c⁺ = contract_twosite(c⁺(; side=:L), c⁺(; side=:R))

    @test cc ≈ -permute(cc, (2, 1), (4, 3))
    @test c⁺c⁺ ≈ -permute(c⁺c⁺, (2, 1), (4, 3))

    # the following doesn't hold
    # I don't think I can get all of these to hold simultaneously?
    # @test cc⁺ ≈ -permute(c⁺c, (2, 1), (4, 3))

    @test cc⁺' ≈ c⁺c
    @test cc' ≈ c⁺c⁺
    @test (c⁺c + cc⁺)' ≈ cc⁺ + c⁺c
    @test (c⁺c - cc⁺)' ≈ cc⁺ - c⁺c

    @test c_number() ≈ contract_onesite(c⁺(; side=:L), c⁻(; side=:R))
end

@testset "electrons $particle_symmetry, $spin_symmetry" for (particle_symmetry, spin_symmetry) in ((Trivial, Trivial), (U1Irrep, SU2Irrep))
    ee⁺ = e_minplus(particle_symmetry, spin_symmetry)
    e⁺e = e_plusmin(particle_symmetry, spin_symmetry)
    @test ee⁺' ≈ e⁺e
    @test (e⁺e + ee⁺)' ≈ ee⁺ + e⁺e
    @test (e⁺e - ee⁺)' ≈ ee⁺ - e⁺e
    
    @test e_number(particle_symmetry, spin_symmetry) ≈ 
        contract_onesite(e⁺(particle_symmetry, spin_symmetry; side=:L), e⁻(particle_symmetry, spin_symmetry; side=:R))
end
