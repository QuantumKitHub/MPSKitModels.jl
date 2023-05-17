using MPSKitModels
using MPSKit
using TensorKit
using Test

E₀ = -1.273239
alg = VUMPS(; maxiter=25)

@testset "single site" begin
    H = transverse_field_ising()
    Ψ₀ = InfiniteMPS([ComplexSpace(2)], [ComplexSpace(16)])
    @test sum(abs.(imag.(expectation_value(Ψ₀, H)))) ≈ 0 atol = 1e-10
    Ψ, envs, δ = find_groundstate(Ψ₀, H, alg)
    @test E₀ ≈ sum(expectation_value(Ψ, H, envs)) atol = 1e-5
end

@testset "Z2 symmetry" begin
    H = transverse_field_ising(ComplexF64, ℤ₂)
    Ψ₀ = InfiniteMPS([Rep[ℤ₂](0 => 1, 1 => 1)], [Rep[ℤ₂](0 => 8, 1 => 8)])
    @test sum(abs.(imag.(expectation_value(Ψ₀, H)))) ≈ 0 atol = 1e-10
    Ψ, envs, δ = find_groundstate(Ψ₀, H, alg)
    @test E₀ ≈ sum(expectation_value(Ψ, H, envs)) atol = 1e-5
end

@testset "fZ2 symmetry" begin
    H = free_fermion_ising(ComplexF64)
    Ψ₀ = InfiniteMPS([Vect[fℤ₂](0 => 1, 1 => 1)], [Vect[fℤ₂](0 => 10, 1 => 10)])
    @test sum(abs.(imag.(expectation_value(Ψ₀, H)))) ≈ 0 atol = 1e-10
    Ψ, envs, δ = find_groundstate(Ψ₀, H, alg)
    @test E₀ ≈ sum(expectation_value(Ψ, H, envs)) atol = 1e-5
end