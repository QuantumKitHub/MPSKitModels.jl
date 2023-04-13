using MPSKitModels
using MPSKit
using Test

const E₀ = -0.318309846883

@testset "single site" begin
    H = transverse_field_ising()
    Ψ₀ = InfiniteMPS(2, 16)
    @test sum(abs.(imag.(expectation_value(Ψ₀, H)))) ≈ 0 atol = 1e-10
    Ψ, envs, δ = find_groundstate(Ψ₀, H; tol=1e-8, maxiter=400, verbose=false)
    @test E₀ ≈ sum(expectation_value(Ψ, H, envs))
end

@testset "double sites" begin
    H2 = transverse_field_ising(ComplexF64, ℤ{1}, InfiniteChain(2))
    Ψ2 = InfiniteMPS([2, 2], [16, 16])

    @test sum(abs.(imag.(expectation_value(Ψ2, H2)))) ≈ 0 atol = 1e-10
    Ψ2, envs, δ = find_groundstate(Ψ2, H2; tol=1e-8, maxiter=400, verbose=false)
    @test 2E₀ ≈ sum(expectation_value(Ψ2, H2, envs))
end

@testset "weird lattice" begin
    lattice = SnakePattern(InfiniteChain(2), i -> iseven(i) ? i - 1 : i + 1)
    H2 = transverse_field_ising(ComplexF64, ℤ{1}, lattice)
    Ψ2 = InfiniteMPS([2, 2], [16, 16])

    @test sum(abs.(imag.(expectation_value(Ψ2, H2)))) ≈ 0 atol = 1e-10
    Ψ2, envs, δ = find_groundstate(Ψ2, H2; tol=1e-8, maxiter=400, verbose=false)
    @test 2E₀ ≈ sum(expectation_value(Ψ2, H2, envs))
end
