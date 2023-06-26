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

@testset "electrons" begin
    ee⁺ = contract_twosite(e⁻(Float64; side=:L), e⁺(Float64; side=:R))
    e⁺e = contract_twosite(e⁺(Float64; side=:L), e⁻(Float64; side=:R))

    @test ee⁺' ≈ e⁺e
    @test (e⁺e + ee⁺)' ≈ ee⁺ + e⁺e
    @test (e⁺e - ee⁺)' ≈ ee⁺ - e⁺e
    
    @test e_number() ≈ contract_onesite(e⁺(; side=:L), e⁻(; side=:R))
    @test e_number_up() + e_number_down() ≈ e_number()
    @test e_number_up() * e_number_down() ≈ e_number_updown()
end
