using MPSKit
using TensorKit

alg = VUMPS(; maxiter=25, verbose=false)
E₀ = -1.401484014561
E₁ = 0.41047925

@testset "xxx" begin
    H = @inferred heisenberg_XXX()
    Ψ = InfiniteMPS([ComplexSpace(3)], [ComplexSpace(48)])
    @test sum(abs.(imag.(expectation_value(Ψ, H)))) ≈ 0 atol = 1e-10
    Ψ, envs, δ = find_groundstate(Ψ, H, alg)
    @test E₀ ≈ sum(expectation_value(Ψ, H, envs)) atol = 1e-2

    ΔEs, qps = excitations(H, QuasiparticleAnsatz(), Float64(pi), Ψ, envs)
    @test E₁ ≈ first(ΔEs) atol = 1e-2
end

@testset "xxx SU2" begin
    H = @inferred heisenberg_XXX(SU2Irrep)
    Ψ = InfiniteMPS([Rep[SU₂](1 => 1)], [Rep[SU₂](1 // 2 => 5, 3 // 2 => 5, 5 // 2 => 1)])
    @test sum(abs.(imag.(expectation_value(Ψ, H)))) ≈ 0 atol = 1e-10
    Ψ, envs, δ = find_groundstate(Ψ, H, alg)
    @test E₀ ≈ sum(expectation_value(Ψ, H, envs)) atol = 1e-2

    ΔEs, qps = excitations(H, QuasiparticleAnsatz(), Float64(pi), Ψ, envs;
                           sector=SU2Irrep(1))
    @test E₁ ≈ first(ΔEs) atol = 1e-2
end

@testset "xxx U1" begin
    H = @inferred heisenberg_XXX(U1Irrep)
    Ψ = InfiniteMPS([Rep[U₁](0 => 1, 1 => 1, -1 => 1)],
                    [Rep[U₁](1 // 2 => 10, -1 // 2 => 10, 3 // 2 => 5, -3 // 2 => 5,
                             5 // 2 => 3, -5 // 2 => 3)])
    @test sum(abs.(imag.(expectation_value(Ψ, H)))) ≈ 0 atol = 1e-10
    Ψ, envs, δ = find_groundstate(Ψ, H, alg)
    @test E₀ ≈ sum(expectation_value(Ψ, H, envs)) atol = 1e-2

    ΔEs, qps = excitations(H, QuasiparticleAnsatz(), Float64(pi), Ψ, envs;
                           sector=U1Irrep(1))
    @test E₁ ≈ first(ΔEs) atol = 1e-2
end
