using MPSKit
using TensorKit
using Test

F₀ = (4 / 3)^(3 / 2)
alg = VUMPS(; maxiter = 25, verbosity = 0)

@testset "ℤ₁" begin
    mpo = @inferred sixvertex()
    ψ = InfiniteMPS(ℂ^2, ℂ^20)
    ψ, _ = leading_boundary(ψ, mpo, alg)
    F = prod(expectation_value(ψ, mpo))
    @test imag(F) < 1.0e-3
    @test real(F) ≈ F₀ atol = 1.0e-2
end

@testset "U₁" begin
    mpo = @inferred sixvertex(ComplexF64, U1Irrep)
    mpo2 = repeat(mpo, 2, 2)
    vspaces = [
        U1Space(0 => 20, 1 => 10, -1 => 10, 2 => 5, -2 => 5),
        U1Space(1 // 2 => 15, -1 // 2 => 15, 3 // 2 => 5, -3 // 2 => 5),
    ]
    ψ = MultilineMPS(repeat(physicalspace(mpo), 2, 2), [vspaces circshift(vspaces, 1)])
    ψ, _ = leading_boundary(ψ, mpo2, alg)
    F = prod(expectation_value(ψ, mpo2))
    @test abs(F)^(1 / 4) ≈ F₀ atol = 1.0e-2
end

@testset "CU₁" begin
    mpo = @inferred sixvertex(ComplexF64, CU1Irrep)
    mpo2 = repeat(mpo, 2, 2)
    vspaces = [
        CU1Space((0, 0) => 10, (0, 1) => 10, (1, 2) => 5, (2, 2) => 5),
        CU1Space((1 // 2, 2) => 15, (3 // 2, 2) => 5),
    ]
    ψ = MultilineMPS(repeat(physicalspace(mpo), 2, 2), [vspaces circshift(vspaces, 1)])
    ψ, _ = leading_boundary(ψ, mpo2, alg)
    F = prod(expectation_value(ψ, mpo2))
    @test abs(F)^(1 / 4) ≈ F₀ atol = 1.0e-2
end
