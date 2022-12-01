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
    Sx = sigma_x(; spin=S)
    Sy = sigma_y(; spin=S)
    Sz = sigma_z(; spin=S)
    
    Svec = [Sx Sy Sz]
    
    # operators should be hermitian
    for s in Svec
        @test s' ≈ s
    end
    
    # operators should be normalized
    @test sum(tr(Svec[i]^2) for i in 1:3) / (2S+1) ≈ S * (S + 1) 
    
    # commutation relations
    for i in 1:3, j in 1:3
        @test Svec[i] * Svec[j] - Svec[j] * Svec[i] ≈
              sum(im * ε[i, j, k] * Svec[k] for k in 1:3)
    end

    # definition of +-
    S⁺ = sigma_plus(; spin=S)
    S⁻ = sigma_min(; spin=S)
    @test (Sx + im * Sy) ≈ S⁺
    @test (Sx - im * Sy) ≈ S⁻
    
    # composite operators
    @test sigma_xx(; spin=S) ≈ Sx ⊗ Sx
    @test sigma_yy(; spin=S) ≈ Sy ⊗ Sy
    @test sigma_zz(; spin=S) ≈ Sz ⊗ Sz
    @test sigma_plusmin(; spin=S) ≈ S⁺ ⊗ S⁻
    @test sigma_minplus(; spin=S) ≈ S⁻ ⊗ S⁺
    @test sigma_exchange(; spin=S) ≈ Sx ⊗ Sx + Sy ⊗ Sy + Sz ⊗ Sz
    @test sigma_exchange(; spin=S) ≈ Sz ⊗ Sz + (S⁺ ⊗ S⁻ + S⁻ ⊗ S⁺) / 2
end
