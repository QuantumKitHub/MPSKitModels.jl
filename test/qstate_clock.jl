using Test
using TensorKit
using QuadGK
using MPSKit
using MPSKitModels

# Onsager free energy per spin of the 2D classical Ising model with coupling J = 1
function classical_ising_free_energy(; beta)
    k = 1 / sinh(2 * beta)^2
    F = quadgk(
        θ -> log(cosh(2 * beta)^2 + 1 / k * sqrt(1 + k^2 - 2 * k * cos(2 * θ))),
        0, pi
    )[1]
    return -1 / beta * (log(2) / 2 + 1 / (2 * pi) * F)
end

alg = VUMPS(; tol = 1.0e-10, maxiter = 100, verbosity = 0)

# Every MPO tensor contains two spins, so the free energy per spin is -log(λ) / (2β).
function clock_free_energy(q, beta; D = 16)
    O = qstate_clock(; beta, q)
    ψ, envs, = leading_boundary(InfiniteMPS(ℂ^q, ℂ^D), O, alg)
    λ = expectation_value(ψ, O, envs)
    return -log(abs(λ)) / (2 * beta)
end

@testset "q = 2 is the Ising model (β = $beta)" for beta in (0.3, 0.6)
    @test clock_free_energy(2, beta) ≈ classical_ising_free_energy(; beta) atol = 1.0e-6
end

# cos(θ - θ') = (σσ' + ττ') / 2 for σ = cos θ + sin θ and τ = cos θ - sin θ, so the q = 4 clock
# model is two decoupled Ising models at half the coupling.
@testset "q = 4 is two Ising models (β = $beta)" for beta in (0.6, 1.2)
    f_ising = classical_ising_free_energy(; beta = beta / 2) * (beta / 2)
    @test clock_free_energy(4, beta) * beta ≈ 2 * f_ising atol = 1.0e-6
end
