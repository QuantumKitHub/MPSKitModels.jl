using MPSKitModels
using MPSKit
using TensorKit
using Test

alg = VUMPS(; maxiter=25)
E₀ = -1.401484014561
E₁ = 0.41047925

@testset "xxx" begin
    H = xxx()
    Ψ = InfiniteMPS([ComplexSpace(3)], [ComplexSpace(48)])
    @test sum(abs.(imag.(expectation_value(Ψ, H)))) ≈ 0 atol = 1e-10
    Ψ, envs, δ = find_groundstate(Ψ, H, alg)
    @test E₀ ≈ sum(expectation_value(Ψ, H, envs)) atol = 1e-2

    ΔEs, qps = excitations(H, QuasiparticleAnsatz(), Float64(pi), Ψ, envs)
    @test E₁ ≈ first(ΔEs) atol = 1e-2
end

@testset "xxx SU2" begin
    H = xxx(ComplexF64, SU₂)
    Ψ = InfiniteMPS([Rep[SU₂](1 => 1)], [Rep[SU₂](1 // 2 => 5, 3 // 2 => 5, 5 // 2 => 1)])
    @test sum(abs.(imag.(expectation_value(Ψ, H)))) ≈ 0 atol = 1e-10
    Ψ, envs, δ = find_groundstate(Ψ, H, alg)
    @test E₀ ≈ sum(expectation_value(Ψ, H, envs)) atol = 1e-2

    ΔEs, qps = excitations(H, QuasiparticleAnsatz(), Float64(pi), Ψ, envs; sector=SU₂(1))
    @test E₁ ≈ first(ΔEs) atol = 1e-2
end

@testset "xxx U1" begin
    H = xxx(ComplexF64, U₁)
    Ψ = InfiniteMPS([Rep[U₁](0 => 1, 1 => 1, -1 => 1)], [Rep[U₁](1//2 => 3, -1//2 => 3, 3//2 => 2, -3//2 => 2)])
    @test sum(abs.(imag.(expectation_value(Ψ, H)))) ≈ 0 atol = 1e-10
    Ψ, envs, δ = find_groundstate(Ψ, H, alg)
    @test E₀ ≈ sum(expectation_value(Ψ, H, envs)) atol = 1e-2

    ΔEs, qps = excitations(H, QuasiparticleAnsatz(), Float64(pi), Ψ, envs; sector=U₁(1))
    @test E₁ ≈ first(ΔEs) atol = 1e-2
end