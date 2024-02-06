using MPSKitModels
using MPSKit
using TensorKit
using Test

E₀ = -1.273239
alg = VUMPS(; maxiter=25, verbose=false)

@testset "no symmetry" begin
    H = @inferred transverse_field_ising()
    ψ₀ = InfiniteMPS([ComplexSpace(2)], [ComplexSpace(16)])
    @test sum(abs.(imag.(expectation_value(ψ₀, H)))) ≈ 0 atol = 1e-10
    ψ, envs, δ = find_groundstate(ψ₀, H, alg)
    @test E₀ ≈ sum(expectation_value(ψ, H, envs)) atol = 1e-5
end

@testset "Z2 symmetry" begin
    H = @inferred transverse_field_ising(Z2Irrep)
    ψ₀ = InfiniteMPS([Rep[ℤ₂](0 => 1, 1 => 1)], [Rep[ℤ₂](0 => 8, 1 => 8)])
    @test sum(abs.(imag.(expectation_value(ψ₀, H)))) ≈ 0 atol = 1e-10
    ψ, envs, δ = find_groundstate(ψ₀, H, alg)
    @test E₀ ≈ sum(expectation_value(ψ, H, envs)) atol = 1e-5
end

@testset "fZ2 symmetry" begin
    H = @inferred transverse_field_ising(fℤ₂)
    ψ₀ = InfiniteMPS([Vect[fℤ₂](0 => 1, 1 => 1)], [Vect[fℤ₂](0 => 8, 1 => 8)])
    @test sum(abs.(imag.(expectation_value(ψ₀, H)))) ≈ 0 atol = 1e-10
    ψ, envs, δ = find_groundstate(ψ₀, H, alg)
    @test E₀ ≈ sum(expectation_value(ψ, H, envs)) atol = 1e-3
end
