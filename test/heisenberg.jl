using MPSKitModels
using MPSKit
using TensorKit
using Test

alg = VUMPS()

@testset "xxx" begin
    E₀ = -1.401484014561
    E₁ = 0.41047925
    H = xxx()
    Ψ = InfiniteMPS([ComplexSpace(3)], [ComplexSpace(48)])
    @test sum(abs.(imag.(expectation_value(Ψ, H)))) ≈ 0 atol = 1e-10
    Ψ, envs, δ = find_groundstate(Ψ, H, alg)
    @test E₀ ≈ sum(expectation_value(Ψ, H, envs)) atol = 1e-4
    
    ΔEs, qps = excitations(H, QuasiparticleAnsatz(), Float64(pi), Ψ, envs)
    @test E₁ ≈ first(ΔEs) atol = 1e-4
end
