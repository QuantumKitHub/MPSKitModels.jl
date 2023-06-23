using MPSKitModels
using TensorKit
using TensorOperations
using Test
using LinearAlgebra: tr

## No symmetry ##
𝕂 = ComplexF64
ε = zeros(𝕂, 3, 3, 3)
for i in 1:3
    ε[mod1(i, 3), mod1(i + 1, 3), mod1(i + 2, 3)] = 1
    ε[mod1(i, 3), mod1(i - 1, 3), mod1(i - 2, 3)] = -1
end

@testset "non-symmetric spin $(Int(2S))/2 operators" for S in (1 // 2):(1 // 2):4
    X = sigma_x(; spin=S)
    Y = sigma_y(; spin=S)
    Z = sigma_z(; spin=S)

    Svec = [X Y Z]

    # operators should be hermitian
    for s in Svec
        @test s' ≈ s
    end

    # operators should be normalized
    @test sum(tr(Svec[i]^2) for i in 1:3) / (2S + 1) ≈ S * (S + 1)

    # commutation relations
    for i in 1:3, j in 1:3
        @test Svec[i] * Svec[j] - Svec[j] * Svec[i] ≈
              sum(im * ε[i, j, k] * Svec[k] for k in 1:3)
    end

    # definition of +-
    S⁺ = sigma_plus(; spin=S)
    S⁻ = sigma_min(; spin=S)
    @test (X + im * Y) ≈ S⁺
    @test (X - im * Y) ≈ S⁻
    @test S⁺' ≈ S⁻

    # composite operators
    @test sigma_xx(; spin=S) ≈ X ⊗ X
    @test sigma_yy(; spin=S) ≈ Y ⊗ Y
    @test sigma_zz(; spin=S) ≈ Z ⊗ Z
    @test sigma_plusmin(; spin=S) ≈ S⁺ ⊗ S⁻
    @test sigma_minplus(; spin=S) ≈ S⁻ ⊗ S⁺
    @test (sigma_plusmin(; spin=S) + sigma_minplus(; spin=S)) / 2 ≈
          sigma_xx(; spin=S) + sigma_yy(; spin=S)
    @test sigma_exchange(; spin=S) ≈ X ⊗ X + Y ⊗ Y + Z ⊗ Z
    @test sigma_exchange(; spin=S) ≈ Z ⊗ Z + (S⁺ ⊗ S⁻ + S⁻ ⊗ S⁺) / 2
end

@testset "Z2-symmetric pauli operators" begin
    # array conversion
    H = [1 1; 1 -1] / sqrt(2)
    @test H * convert(Array, sigma_x()) * H' ≈ convert(Array, sigma_x(Z2Irrep))
    for sigma in (sigma_y, sigma_z, sigma_plus, sigma_min)
        array1 = H * convert(Array, sigma()) * H'
        arrayL = reshape(sum(convert(Array, sigma(Z2Irrep; side=:L)); dims=3), 2, 2)
        arrayR = reshape(sum(convert(Array, sigma(Z2Irrep; side=:R)); dims=1), 2, 2)
        @test array1 ≈ arrayL
        @test array1 ≈ arrayR
    end

    # hermiticity
    @test sigma_x(Z2Irrep)' ≈ sigma_x(Z2Irrep)
    @test permute(sigma_y(Z2Irrep; side=:L)', (2, 1), (3,)) ≈ sigma_y(Z2Irrep; side=:R)
    @test permute(sigma_z(Z2Irrep; side=:L)', (2, 1), (3,)) ≈ sigma_z(Z2Irrep; side=:R)
    @test permute(sigma_plus(Z2Irrep; side=:L)', (2, 1), (3,)) ≈ sigma_min(Z2Irrep; side=:R)
    @test permute(sigma_min(Z2Irrep; side=:L)', (2, 1), (3,)) ≈ sigma_plus(Z2Irrep; side=:R)

    # composite operators
    @test (sigma_plusmin(Z2Irrep) + sigma_minplus(Z2Irrep)) / 2 ≈
          sigma_xx(Z2Irrep) + sigma_yy(Z2Irrep) rtol = 1e-3
end

@testset "U1-symmetric spin $(Int(2S))/2 operators" for S in (1 // 2):(1 // 2):4
    # array conversion
    N = Int(2S + 1)
    p = sortperm((-S):S; by=x -> abs(x - 0.1)) # sort as 0, 1, -1, 2, -2, ...
    H = one(zeros(N, N))[p, :]
    @test H * convert(Array, sigma_z(; spin=S)) * H' ≈
          convert(Array, sigma_z(U1Irrep; spin=S))
    for sigma in (sigma_x, sigma_y, sigma_plus, sigma_min)
        array1 = convert(Array, sigma(; spin=S))
        arrayL = H' * reshape(sum(convert(Array, sigma(U1Irrep; side=:L, spin=S)); dims=3), N, N) * H
        arrayR = H' * reshape(sum(convert(Array, sigma(U1Irrep; side=:R, spin=S)); dims=1), N, N) * H
        @test array1 ≈ arrayL
        @test array1 ≈ arrayR
    end

    # # hermiticity
    @test sigma_z(U1Irrep; spin=S)' ≈ sigma_z(U1Irrep; spin=S)
    @test permute(sigma_x(U1Irrep; spin=S, side=:L)', (2, 1), (3,)) ≈ sigma_x(U1Irrep; spin=S, side=:R)
    @test permute(sigma_y(U1Irrep; spin=S, side=:L)', (2, 1), (3,)) ≈ sigma_y(U1Irrep; spin=S, side=:R)
    @test permute(sigma_plus(U1Irrep; spin=S, side=:L)', (2, 1), (3,)) ≈ sigma_min(U1Irrep; spin=S, side=:R)
    @test permute(sigma_min(U1Irrep; spin=S, side=:L)', (2, 1), (3,)) ≈ sigma_plus(U1Irrep; spin=S, side=:R)

    # # composite operators
    @test (sigma_plusmin(U1Irrep; spin=S) + sigma_minplus(U1Irrep; spin=S)) / 2 ≈
          sigma_xx(U1Irrep; spin=S) + sigma_yy(U1Irrep; spin=S) rtol = 1e-3
    @test sigma_exchange(U1Irrep; spin=S) ≈
          sigma_xx(U1Irrep; spin=S) + sigma_yy(U1Irrep; spin=S) + sigma_zz(U1Irrep; spin=S) rtol = 1e-3
end
