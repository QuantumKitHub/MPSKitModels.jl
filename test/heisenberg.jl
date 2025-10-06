using MPSKit
using TensorKit

alg = VUMPS(; maxiter = 25, verbosity = 0)
E₀ = -1.401484014561
E₁ = 0.41047925

@testset "xxx" begin
    H = heisenberg_XXX()
    ψ = InfiniteMPS([ComplexSpace(3)], [ComplexSpace(48)])
    @test imag(expectation_value(ψ, H)) ≈ 0 atol = 1.0e-10
    ψ, envs, δ = find_groundstate(ψ, H, alg)
    @test E₀ ≈ expectation_value(ψ, H, envs) atol = 1.0e-2

    ΔEs, qps = excitations(H, QuasiparticleAnsatz(), Float64(pi), ψ, envs)
    @test E₁ ≈ first(ΔEs) atol = 1.0e-2
end

@testset "xxx SU2" begin
    H = heisenberg_XXX(SU2Irrep)
    ψ = InfiniteMPS([Rep[SU₂](1 => 1)], [Rep[SU₂](1 // 2 => 5, 3 // 2 => 5, 5 // 2 => 1)])
    @test imag(expectation_value(ψ, H)) ≈ 0 atol = 1.0e-10
    ψ, envs, δ = find_groundstate(ψ, H, alg)
    @test E₀ ≈ expectation_value(ψ, H, envs) atol = 1.0e-2

    ΔEs, qps = excitations(
        H, QuasiparticleAnsatz(), Float64(pi), ψ, envs;
        sector = SU2Irrep(1)
    )
    @test E₁ ≈ first(ΔEs) atol = 1.0e-2
end

@testset "xxx U1" begin
    H = heisenberg_XXX(U1Irrep)
    ψ = InfiniteMPS(
        [Rep[U₁](0 => 1, 1 => 1, -1 => 1)],
        [
            Rep[U₁](
                1 // 2 => 10, -1 // 2 => 10, 3 // 2 => 5, -3 // 2 => 5,
                5 // 2 => 3, -5 // 2 => 3
            ),
        ]
    )
    @test imag(expectation_value(ψ, H)) ≈ 0 atol = 1.0e-10
    ψ, envs, δ = find_groundstate(ψ, H, alg)
    @test E₀ ≈ expectation_value(ψ, H, envs) atol = 1.0e-2

    ΔEs, qps = excitations(
        H, QuasiparticleAnsatz(), Float64(pi), ψ, envs;
        sector = U1Irrep(1)
    )
    @test E₁ ≈ first(ΔEs) atol = 1.0e-2
end
