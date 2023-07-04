using MPSKit
using TensorKit
using Test

F₀ = (4/3)^(3/2)
alg = VUMPS(; maxiter=25, verbose=false)

@testset "ℤ₁" begin
    mpo = sixvertex()
    ψ = InfiniteMPS(ℂ^2, ℂ^20)
    ψ, _ = leading_boundary(ψ, mpo, alg)
    F = prod(expectation_value(ψ, mpo))
    @test imag(F) < 1e-3
    @test real(F) ≈ F₀ atol = 1e-2
end

@testset "U₁" begin
    mpo = sixvertex(ComplexF64, U1Irrep)
    mpo2 = MPOMultiline([mpo.opp mpo.opp; mpo.opp mpo.opp])
    vspaces = [U1Space(0 => 20, 1 => 10, -1 => 10, 2 => 5, -2 => 5), U1Space(1 // 2 => 15, -1//2 => 15, 3//2 => 5, -3 //2 => 5)]
    ψ = MPSMultiline(repeat(space.(mpo.opp, 2), 2, 2), [vspaces circshift(vspaces, 1)])
    ψ, _ = leading_boundary(ψ, mpo2, alg)
    F = prod(expectation_value(ψ, mpo2))
    @test abs(F) ^ (1/4) ≈ F₀ atol = 1e-2
end

@testset "CU₁" begin
    mpo = sixvertex(ComplexF64, CU1Irrep)
    mpo2 = MPOMultiline([mpo.opp mpo.opp; mpo.opp mpo.opp])
    vspaces = [CU1Space((0, 0) => 10, (0, 1) => 10, (1, 2) => 5, (2, 2) => 5),
               CU1Space((1 // 2, 2) => 15, (3 // 2, 2) => 5)]
    ψ = MPSMultiline(repeat(space.(mpo.opp, 2), 2, 2), [vspaces circshift(vspaces, 1)])
    ψ, _ = leading_boundary(ψ, mpo2, alg)
    F = prod(expectation_value(ψ, mpo2))
    @test abs(F)^(1 / 4) ≈ F₀ atol = 1e-2
end