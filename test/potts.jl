using MPSKit
using TensorKit

alg = VUMPS(; maxiter=25, verbosity=0)
E₀ = -(4 / 3 + 2sqrt(3) / π)

@testset "Trivial" begin
    H = quantum_potts(; q=3)
    ψ = InfiniteMPS(3, 32)
    @test imag(expectation_value(ψ, H)) ≈ 0 atol = 1e-10
    ψ, envs, δ = find_groundstate(ψ, H, alg)
    @test E₀ ≈ expectation_value(ψ, H, envs) atol = 1e-2
end

@testset "Z3Irrep" begin
    H = quantum_potts(Z3Irrep; q=3)
    ψ = InfiniteMPS(Rep[ℤ₃](i => 1 for i in 0:2), Rep[ℤ₃](i => 12 for i in 0:2))
    @test imag(expectation_value(ψ, H)) ≈ 0 atol = 1e-10
    ψ, envs, δ = find_groundstate(ψ, H, alg)
    @test E₀ ≈ expectation_value(ψ, H, envs) atol = 1e-2
end
