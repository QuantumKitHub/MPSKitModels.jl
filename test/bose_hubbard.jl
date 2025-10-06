using Test
using TensorKit
using MPSKit
using MPSKitModels

## Setup

symmetry = U1Irrep
cutoff = 3
t = 1.0
U = 10.0
mu = 0.0
n = 1
lattice = InfiniteChain()

Vspace = U1Space(0 => 8, 1 => 6, -1 => 6, 2 => 4, -2 => 4, 3 => 2, -3 => 2)

alg = VUMPS(; maxiter = 25, verbosity = 0)

# compare against higher-order analytic expansion from https://arxiv.org/pdf/1507.06426
function exact_bose_hubbard_energy(; t = 1.0, U = 1.0)
    J = t / U

    E = 4 * U *
        (
        -J^2 + J^4 + 68 / 9 * J^6 - 1267 / 81 * J^8 + 44171 / 1458 * J^10 -
            4902596 / 6561 * J^12 -
            8020902135607 / 2645395200 * J^14 - 32507578587517774813 / 466647713280000 * J^16
    )

    return E
end

## Test

@testset "Bose-Hubbard ground state" begin
    H = bose_hubbard_model(symmetry, lattice; cutoff, t, U, mu, n)
    ψ = InfiniteMPS([physicalspace(H, 1)], [Vspace])
    @test imag(expectation_value(ψ, H)) ≈ 0 atol = 1.0e-10
    ψ, envs, δ = find_groundstate(ψ, H, alg)
    @test expectation_value(ψ, H, envs) ≈ exact_bose_hubbard_energy(; t, U) atol = 1.0e-2
end
