using MPSKit
using TensorKit

alg = VUMPS(; maxiter=25, verbosity=0)

# https://iopscience.iop.org/article/10.1088/0305-4470/14/11/020/meta
function E₀(Q::Int, maxiter::Int = 1000)
    Q == 3 && return -(4 / 3 + 2sqrt(3) / π)
    Q == 4 && return 2 - 8*log(2)
    summation = sum((-1)^n / (sqrt(Q)/2 - cosh((2*n+1)*acosh(sqrt(Q)/2))) for n in 1:maxiter)
    limit = 2 - Q - sqrt(Q)*(Q-4)*summation
    return limit
end

# Q = 3
@testset "Q=3 Trivial" begin
    H = quantum_potts(; q=3)
    ψ = InfiniteMPS(3, 32)
    @test imag(expectation_value(ψ, H)) ≈ 0 atol = 1e-10
    ψ, envs, δ = find_groundstate(ψ, H, alg)
    @test E₀(3) ≈ expectation_value(ψ, H, envs) atol = 1e-2
end

@testset "Z3Irrep" begin
    H = quantum_potts(Z3Irrep; q=3)
    ψ = InfiniteMPS(Rep[ℤ₃](i => 1 for i in 0:2), Rep[ℤ₃](i => 12 for i in 0:2))
    @test imag(expectation_value(ψ, H)) ≈ 0 atol = 1e-10
    ψ, envs, δ = find_groundstate(ψ, H, alg)
    @test E₀(3) ≈ expectation_value(ψ, H, envs) atol = 1e-2
end

# Q = 4
@testset "Q=4 Trivial" begin
    H = quantum_potts(; q=4)
    ψ = InfiniteMPS(4, 45)
    @test imag(expectation_value(ψ, H)) ≈ 0 atol = 1e-10
    ψ, envs, δ = find_groundstate(ψ, H, alg)
    @test E₀(4) ≈ expectation_value(ψ, H, envs) atol = 1e-2
end

@testset "Z4Irrep" begin
    H = quantum_potts(Z4Irrep; q=4)
    ψ =  InfiniteMPS(Vect[Z4Irrep](i => 1 for i in 0:3), Vect[Z4Irrep](i => 12 for i in 0:3));
    @test imag(expectation_value(ψ, H)) ≈ 0 atol = 1e-10
    ψ, envs, δ = find_groundstate(ψ, H, alg)
    @test E₀(4) ≈ expectation_value(ψ, H, envs) atol = 1e-2
end

# Q = 5
@testset "Q=5 Trivial" begin
    H = quantum_potts(; q=5)
    ψ = InfiniteMPS(5, 60)
    @test imag(expectation_value(ψ, H)) ≈ 0 atol = 1e-10
    ψ, envs, δ = find_groundstate(ψ, H, alg)
    @test E₀(5) ≈ expectation_value(ψ, H, envs) atol = 1e-2
end

@testset "ZNIrrep{5}" begin
    H = quantum_potts(ZNIrrep{5}; q=5)
    ψ =  InfiniteMPS(Vect[ZNIrrep{5}](i => 1 for i in 0:4), Vect[ZNIrrep{5}](i => 12 for i in 0:4));
    @test imag(expectation_value(ψ, H)) ≈ 0 atol = 1e-10
    ψ, envs, δ = find_groundstate(ψ, H, alg)
    @test E₀(5) ≈ expectation_value(ψ, H, envs) atol = 1e-2
end
