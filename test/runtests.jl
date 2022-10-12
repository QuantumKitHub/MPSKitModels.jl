using Test,TensorKit,MPSKit,MPSKitModels

@testset "xxz" begin
    @testset "nonsym_xxz_ham" begin
        th = nonsym_xxz_ham()
        ts = InfiniteMPS([ℂ^3],[ℂ^48]);
        (ts,pars,_) = find_groundstate(ts,th,VUMPS(maxiter=400,verbose=false));
        (energies,Bs) = excitations(th,QuasiparticleAnsatz(),Float64(pi),ts,pars);
        @test energies[1] ≈ 0.41047925 atol=1e-4
    end
end
