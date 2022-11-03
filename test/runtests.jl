using Test, TensorKit, MPSKit, MPSKitModels

@testset "infinite systems" begin
    _, _, Sz = nonsym_spintensors(1 // 2)
    @macroexpand @mpoham sum([Sz{i} for i in -Inf:Inf])
    H = @mpoham sum([Sz{i} for i in -Inf:Inf])
    J = 0.5
    H2 = @mpoham sum([J * Sz{i} for i in -Inf:Inf])
    J1 = 0.1
    J2 = 0.3
    H3 = @mpoham sum([J1 * Sz{i} + J2 * Sz{i + 1} for i in -Inf:2:Inf])
    J3 = [0.1 0.2 0.3 0.4]
    H4 = @mpoham sum([J3[i] * Sz{i} for i in -∞:4:∞])
end



@testset "xxz" begin
    @testset "nonsym_xxz_ham" begin
        th = nonsym_xxz_ham()
        ts = InfiniteMPS([ℂ^3], [ℂ^48])
        (ts, pars, _) = find_groundstate(ts, th, VUMPS(maxiter=400, verbose=false))
        (energies, Bs) = excitations(th, QuasiparticleAnsatz(), Float64(pi), ts, pars)
        @test energies[1] ≈ 0.41047925 atol = 1e-4
    end
end
