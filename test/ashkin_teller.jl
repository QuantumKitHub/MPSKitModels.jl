using Test
using TensorKit
using MPSKit
using MPSKitModels

# Setup

gammas = [
    pi / 2, pi / 3, pi / 4, pi / 6, 0,
]
# Exact GS energy density via mapping to spin 1/2 XXZ and Bethe ansatz
# https://www.sciencedirect.com/science/article/pii/0003491688900152
E0s = [
    -8 / pi, -12 / 4, -4 * (sqrt(2) / pi + 1 / (2 * sqrt(2))),
    -4 * (1 / pi + 11 / (12 * sqrt(3))), 2 - 8 * log(2),
]
alg = VUMPS(; maxiter = 100, verbosity = 0)

# Test

@testset "Ashkin-Teller" for (gamma, E0) in zip(gammas, E0s)
    H = ashkin_teller(h = 1, J = 1, λ = cos(gamma))
    Ψ = InfiniteMPS(physicalspace(H), [ComplexSpace(24)])
    @test imag(expectation_value(Ψ, H)) ≈ 0 atol = 1.0e-12
    Ψ0, _ = find_groundstate(Ψ, H, alg)
    @test real(expectation_value(Ψ0, H)) ≈ E0 atol = 1.0e-3
end

@testset "Ashkin-Teller (Z2 ⊠ Z2)" for (gamma, E0) in zip(gammas, E0s)
    H = ashkin_teller(Z2Irrep ⊠ Z2Irrep; h = 1, J = 1, λ = cos(gamma))
    V = spacetype(H)(c => 6 for c in values(sectortype(H)))
    Ψ = InfiniteMPS(physicalspace(H), [V])
    @test imag(expectation_value(Ψ, H)) ≈ 0 atol = 1.0e-12
    Ψ0, _ = find_groundstate(Ψ, H, alg)
    @test real(expectation_value(Ψ0, H)) ≈ E0 atol = 1.0e-3
end
