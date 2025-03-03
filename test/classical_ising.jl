using Test
using TensorKit
using QuadGK
using MPSKit
using MPSKitModels

## Setup

beta = 0.6
Vspaces = [ComplexSpace(12), Z2Space(0 => 6, 1 => 6)]
alg = VUMPS(; tol=1e-8, maxiter=25, verbosity=1)

"""
    classical_ising_free_energy(; beta)

[Exact Onsager solution](https://en.wikipedia.org/wiki/Square_lattice_Ising_model#Exact_solution)
for the free energy of the 2D classical Ising Model with partition function

```math
\\mathcal{Z}(\\beta) = \\sum_{\\{s\\}} \\exp(-\\beta H(s)) \\text{ with } H(s) = - \\sum_{\\langle i, j \\rangle} s_i s_j
```
"""
function classical_ising_free_energy(; beta=log(1 + sqrt(2)) / 2)
    k = 1 / sinh(2 * beta)^2
    F = quadgk(theta -> log(cosh(2 * beta)^2 +
                            1 / k * sqrt(1 + k^2 - 2 * k * cos(2 * theta))),
               0, pi)[1]
    return -1 / beta * (log(2) / 2 + 1 / (2 * pi) * F)
end

## Test

@testset "Classical Ising for $(sectortype(V)) symmetry" for V in Vspaces
    O = classical_ising(sectortype(V); beta)
    psi0 = InfiniteMPS(physicalspace(O, 1), V)
    psi, envs, = leading_boundary(psi0, O, alg)
    λ = expectation_value(psi, O, envs)
    f = -log(λ) / beta
    f_exact = classical_ising_free_energy(; beta)
    @test abs(f - f_exact) < 1e-4
end
