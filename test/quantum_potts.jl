using Test
using TensorKit
using MPSKit
using MPSKitModels

## Setup

χ = 6
qs = [3, 4, 5]
symmetries = [Trivial, ZNIrrep]
Vspaces = [ComplexSpace(12), Z2Space(0 => 6, 1 => 6)]
alg = VUMPS(; maxiter = 25, verbosity = 0)

# https://iopscience.iop.org/article/10.1088/0305-4470/14/11/020/meta
function quantum_potts_energy(Q::Int; maxiter::Int = 1000)
    Q == 3 && return -(4 / 3 + 2sqrt(3) / π)
    Q == 4 && return 2 - 8 * log(2)
    summation = sum(
        (-1)^n / (sqrt(Q) / 2 - cosh((2 * n + 1) * acosh(sqrt(Q) / 2)))
            for n in 1:maxiter
    )
    limit = 2 - Q - sqrt(Q) * (Q - 4) * summation
    return limit
end

_sectortype(::Type{Trivial}, q::Int) = Trivial
_sectortype(::Type{ZNIrrep}, q::Int) = ZNIrrep{q}
_virtualspace(::Type{Trivial}, q::Int, χ::Int) = ComplexSpace(q * χ)
_virtualspace(::Type{ZNIrrep}, q::Int, χ::Int) = Rep[ℤ{q}](i => χ for i in 0:(q - 1))

## Test

@testset "$q-state Potts with $(_sectortype(symmetry, q))) symmetry" for q in qs, symmetry in symmetries
    H = quantum_potts(_sectortype(symmetry, q); q)
    ψ = InfiniteMPS(physicalspace(H, 1), _virtualspace(symmetry, q, χ))
    @test imag(expectation_value(ψ, H)) ≈ 0 atol = 1.0e-10
    ψ, envs, δ = find_groundstate(ψ, H, alg)
    @test quantum_potts_energy(q) ≈ expectation_value(ψ, H, envs) atol = 1.0e-2
end
