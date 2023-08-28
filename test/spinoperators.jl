using TensorKit
using TensorOperations
using LinearAlgebra: tr

## No symmetry ##
𝕂 = ComplexF64
ε = zeros(𝕂, 3, 3, 3)
for i in 1:3
    ε[mod1(i, 3), mod1(i + 1, 3), mod1(i + 2, 3)] = 1
    ε[mod1(i, 3), mod1(i - 1, 3), mod1(i - 2, 3)] = -1
end

@testset "non-symmetric spin $(Int(2S))/2 operators" for S in (1 // 2):(1 // 2):4
    X = S_x(; spin=S)
    Y = S_y(; spin=S)
    Z = S_z(; spin=S)

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
    S⁺ = S_plus(; spin=S)
    S⁻ = S_min(; spin=S)
    @test (X + im * Y) ≈ S⁺
    @test (X - im * Y) ≈ S⁻
    @test S⁺' ≈ S⁻

    # composite operators
    @test S_xx(; spin=S) ≈ X ⊗ X
    @test S_yy(; spin=S) ≈ Y ⊗ Y
    @test S_zz(; spin=S) ≈ Z ⊗ Z
    @test S_plusmin(; spin=S) ≈ S⁺ ⊗ S⁻
    @test S_minplus(; spin=S) ≈ S⁻ ⊗ S⁺
    @test (S_plusmin(; spin=S) + S_minplus(; spin=S)) / 2 ≈
          S_xx(; spin=S) + S_yy(; spin=S)
    @test S_exchange(; spin=S) ≈ X ⊗ X + Y ⊗ Y + Z ⊗ Z
    @test S_exchange(; spin=S) ≈ Z ⊗ Z + (S⁺ ⊗ S⁻ + S⁻ ⊗ S⁺) / 2
end

@testset "Z2-symmetric pauli operators" begin
    # array conversion
    H = [1 1; 1 -1] / sqrt(2)
    @test H * convert(Array, S_x()) * H' ≈ convert(Array, S_x(Z2Irrep))
    for S in (S_y, S_z, S_plus, S_min)
        array1 = H * convert(Array, S()) * H'
        arrayL = reshape(sum(convert(Array, S(Z2Irrep; side=:L)); dims=3), 2, 2)
        arrayR = reshape(sum(convert(Array, S(Z2Irrep; side=:R)); dims=1), 2, 2)
        @test array1 ≈ arrayL
        @test array1 ≈ arrayR
    end

    # hermiticity
    @test S_x(Z2Irrep)' ≈ S_x(Z2Irrep)
    @test permute(S_y(Z2Irrep; side=:L)', ((2, 1), (3,))) ≈ S_y(Z2Irrep; side=:R)
    @test permute(S_z(Z2Irrep; side=:L)', ((2, 1), (3,))) ≈ S_z(Z2Irrep; side=:R)
    @test permute(S_plus(Z2Irrep; side=:L)', ((2, 1), (3,))) ≈ S_min(Z2Irrep; side=:R)
    @test permute(S_min(Z2Irrep; side=:L)', ((2, 1), (3,))) ≈ S_plus(Z2Irrep; side=:R)

    # composite operators
    @test (S_plusmin(Z2Irrep) + S_minplus(Z2Irrep)) / 2 ≈
          S_xx(Z2Irrep) + S_yy(Z2Irrep) rtol = 1e-3
end

@testset "U1-symmetric spin $(Int(2spin))/2 operators" for spin in (1 // 2):(1 // 2):4
    # array conversion
    N = Int(2spin + 1)
    p = sortperm((-spin):spin; by=x -> abs(x - 0.1)) # sort as 0, 1, -1, 2, -2, ...
    H = one(zeros(N, N))[p, :]
    @test H * convert(Array, S_z(; spin=spin)) * H' ≈
          convert(Array, S_z(U1Irrep; spin=spin))
    for S in (S_x, S_y, S_plus, S_min)
        array1 = convert(Array, S(; spin=spin))
        arrayL = H' *
                 reshape(sum(convert(Array, S(U1Irrep; side=:L, spin=spin)); dims=3), N,
                         N) * H
        arrayR = H' *
                 reshape(sum(convert(Array, S(U1Irrep; side=:R, spin=spin)); dims=1), N,
                         N) * H
        @test array1 ≈ arrayL
        @test array1 ≈ arrayR
    end

    # # hermiticity
    @test S_z(U1Irrep; spin=spin)' ≈ S_z(U1Irrep; spin=spin)
    @test permute(S_x(U1Irrep; spin=spin, side=:L)', ((2, 1), (3,))) ≈
          S_x(U1Irrep; spin=spin, side=:R)
    @test permute(S_y(U1Irrep; spin=spin, side=:L)', ((2, 1), (3,))) ≈
          S_y(U1Irrep; spin=spin, side=:R)
    @test permute(S_plus(U1Irrep; spin=spin, side=:L)', ((2, 1), (3,))) ≈
          S_min(U1Irrep; spin=spin, side=:R)
    @test permute(S_min(U1Irrep; spin=spin, side=:L)', ((2, 1), (3,))) ≈
          S_plus(U1Irrep; spin=spin, side=:R)

    # # composite operators
    @test (S_plusmin(U1Irrep; spin=spin) + S_minplus(U1Irrep; spin=spin)) / 2 ≈
          S_xx(U1Irrep; spin=spin) + S_yy(U1Irrep; spin=spin) rtol = 1e-3
    @test S_exchange(U1Irrep; spin=spin) ≈
          S_xx(U1Irrep; spin=spin) + S_yy(U1Irrep; spin=spin) + S_zz(U1Irrep; spin=spin) rtol = 1e-3
end
