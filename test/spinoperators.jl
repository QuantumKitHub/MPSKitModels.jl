using TensorKit
using TensorOperations
using LinearAlgebra: tr, I
using TestExtras

## No symmetry ##
𝕂 = ComplexF64
ε = zeros(𝕂, 3, 3, 3)
for i in 1:3
    ε[mod1(i, 3), mod1(i + 1, 3), mod1(i + 2, 3)] = 1
    ε[mod1(i, 3), mod1(i - 1, 3), mod1(i - 2, 3)] = -1
end

@testset "non-symmetric spin $(Int(2S))/2 operators" for S in (1 // 2):(1 // 2):4
    # inferrability
    X = @inferred S_x(; spin=S)
    Y = @inferred S_y(; spin=S)
    Z = @inferred S_z(; spin=S)
    S⁺ = @inferred S_plus(; spin=S)
    S⁻ = @inferred S_min(; spin=S)
    S⁺⁻ = @inferred S_plusmin(; spin=S)
    S⁻⁺ = @inferred S_minplus(; spin=S)
    XX = @inferred S_xx(; spin=S)
    YY = @inferred S_yy(; spin=S)
    ZZ = @inferred S_zz(; spin=S)
    SS = @inferred S_exchange(; spin=S)
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
    @test (X + im * Y) ≈ S⁺
    @test (X - im * Y) ≈ S⁻
    @test S⁺' ≈ S⁻

    # composite operators
    @test XX ≈ X ⊗ X
    @test YY ≈ Y ⊗ Y
    @test ZZ ≈ Z ⊗ Z
    @test S⁺⁻ ≈ S⁺ ⊗ S⁻
    @test S⁻⁺ ≈ S⁻ ⊗ S⁺
    @test (S⁺⁻ + S⁻⁺) / 2 ≈ XX + YY
    @test SS ≈ X ⊗ X + Y ⊗ Y + Z ⊗ Z
    @test SS ≈ Z ⊗ Z + (S⁺ ⊗ S⁻ + S⁻ ⊗ S⁺) / 2
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

    # inferrability
    X = @inferred S_x(Z2Irrep)
    YL = @constinferred S_y(Z2Irrep; side=:L)
    YR = @constinferred S_y(Z2Irrep; side=:R)
    ZL = @constinferred S_z(Z2Irrep; side=:L)
    ZR = @constinferred S_z(Z2Irrep; side=:R)
    S⁺L = @constinferred S_plus(Z2Irrep; side=:L)
    S⁺R = @constinferred S_plus(Z2Irrep; side=:R)
    S⁻L = @constinferred S_min(Z2Irrep; side=:L)
    S⁻R = @constinferred S_min(Z2Irrep; side=:R)
    S⁺⁻ = @inferred S_plusmin(Z2Irrep)
    S⁻⁺ = @inferred S_minplus(Z2Irrep)
    XX = @inferred S_xx(Z2Irrep)
    YY = @inferred S_yy(Z2Irrep)

    # hermiticity
    @test X' ≈ X
    @test permute(YL', ((2, 1), (3,))) ≈ YR
    @test permute(ZL', ((2, 1), (3,))) ≈ ZR
    @test permute(S⁺L', ((2, 1), (3,))) ≈ S⁻R
    @test permute(S⁻L', ((2, 1), (3,))) ≈ S⁺R

    # composite operators
    @test (S⁺⁻ + S⁻⁺) / 2 ≈ XX + YY rtol = 1e-3
end

@testset "U1-symmetric spin $(Int(2spin))/2 operators" for spin in (1 // 2):(1 // 2):4
    # array conversion
    N = Int(2spin + 1)
    p = sortperm(reverse((-spin):spin); by=x -> abs(x - 0.1)) # sort as 0, 1, -1, 2, -2, ...
    H = one(zeros(N, N))[p, :]
    @test H * convert(Array, S_z(; spin=spin)) * H' ≈
          convert(Array, S_z(U1Irrep; spin=spin))
    for S in (S_x, S_y, S_plus, S_min)
        array1 = convert(Array, S(; spin=spin))
        arrayL = H' *
                 reshape(sum(convert(Array, S(U1Irrep; side=:L, spin=spin)); dims=3), N,
                         N) *
                 H
        arrayR = H' *
                 reshape(sum(convert(Array, S(U1Irrep; side=:R, spin=spin)); dims=1), N,
                         N) *
                 H
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

@testset "Real eltype" begin
    for operator in (S_x, S_z, S_xx, S_zz, S_plusmin, S_minplus, S_exchange),
        symmetry in (Trivial, Z2Irrep, U1Irrep)
        @test real(operator(ComplexF64, symmetry)) ≈ operator(Float64, symmetry)
    end
    @test real(S_exchange(ComplexF64, SU2Irrep)) ≈ S_exchange(Float64, SU2Irrep)
end

# potts_ZZ test?
@testset "non-symmetric Q-state potts operators" for Q in 3:5
    # inferrability
    X = @inferred potts_X(; q=Q)
    Z = @inferred potts_Z(; q=Q)
    ZZ = @inferred potts_ZZ(; q=Q)

    # clock properties
    @test convert(Array, X^Q) ≈ I
    @test convert(Array, Z^Q) ≈ I

    # dagger should be reversing the clock direction
    for s in [X Z]
        for i in 1:Q
            @test (s')^i ≈ s^(Q - i)
        end
    end

    # commutation relations
    ω = cis(2π / Q)
    @test Z * X ≈ ω * X * Z
end

# potts_ZZ test?
@testset "Z_Q-symmetric Q-state Potts operators" for Q in 3:5
    # array conversion
    _, _, W = weyl_heisenberg_matrices(Q, ComplexF64)
    @test W * convert(Array, potts_X(; q=Q)) * W' ≈ convert(Array, potts_X(ZNIrrep{Q}; q=Q))

    # inferrability
    X = @inferred potts_X(ZNIrrep{Q}; q=Q)
    ZZ = @inferred potts_ZZ(ZNIrrep{Q}; q=Q)

    # unitarity
    @test X * X' ≈ X' * X
    @test convert(Array, X * X') ≈ I
    @test convert(Array, X^Q) ≈ I

    # dagger should be reversing the clock direction
    for i in 1:Q
        @test (X')^i ≈ X^(Q - i)
    end
end
