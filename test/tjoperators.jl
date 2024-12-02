using Test
using TensorKit
using LinearAlgebra: eigvals
import MPSKitModels: TJOperators as tJ

implemented_symmetries = [(Trivial, Trivial),
                          (Trivial, U1Irrep),
                          (U1Irrep, Trivial),
                          (U1Irrep, U1Irrep)]
@testset "basic properties" begin
    for slave_fermion in (false, true),
        particle_symmetry in (Trivial, U1Irrep),
        spin_symmetry in (Trivial, U1Irrep, SU2Irrep)

        if (particle_symmetry, spin_symmetry) in implemented_symmetries
            # test hermiticity
            @test tJ.e_plusmin(particle_symmetry, spin_symmetry; slave_fermion)' ≈
                  tJ.e_minplus(particle_symmetry, spin_symmetry; slave_fermion)
            if spin_symmetry !== SU2Irrep
                @test tJ.e_plusmin_down(particle_symmetry, spin_symmetry; slave_fermion)' ≈
                      tJ.e_minplus_down(particle_symmetry, spin_symmetry; slave_fermion)
                @test tJ.e_plusmin_up(particle_symmetry, spin_symmetry; slave_fermion)' ≈
                      tJ.e_minplus_up(particle_symmetry, spin_symmetry; slave_fermion)
                @test tJ.e_plusmin_down(particle_symmetry, spin_symmetry; slave_fermion)' ≈
                      tJ.e_minplus_down(particle_symmetry, spin_symmetry; slave_fermion)
                @test tJ.e_plusmin_up(particle_symmetry, spin_symmetry; slave_fermion)' ≈
                      tJ.e_minplus_up(particle_symmetry, spin_symmetry; slave_fermion)
            end

            # test number operator
            if spin_symmetry !== SU2Irrep
                pspace = tJ.tj_space(particle_symmetry, spin_symmetry; slave_fermion)
                @test tJ.e_number(particle_symmetry, spin_symmetry; slave_fermion) ≈
                      tJ.e_number_up(particle_symmetry, spin_symmetry; slave_fermion) +
                      tJ.e_number_down(particle_symmetry, spin_symmetry; slave_fermion)
                @test TensorMap(zeros, pspace, pspace) ≈
                      tJ.e_number_up(particle_symmetry, spin_symmetry; slave_fermion) *
                      tJ.e_number_down(particle_symmetry, spin_symmetry; slave_fermion) ≈
                      tJ.e_number_down(particle_symmetry, spin_symmetry; slave_fermion) *
                      tJ.e_number_up(particle_symmetry, spin_symmetry; slave_fermion)
            end

            # test spin operator
            if spin_symmetry == Trivial
                ε = zeros(ComplexF64, 3, 3, 3)
                for i in 1:3
                    ε[mod1(i, 3), mod1(i + 1, 3), mod1(i + 2, 3)] = 1
                    ε[mod1(i, 3), mod1(i - 1, 3), mod1(i - 2, 3)] = -1
                end
                Svec = [tJ.S_x(particle_symmetry, spin_symmetry; slave_fermion),
                        tJ.S_y(particle_symmetry, spin_symmetry; slave_fermion),
                        tJ.S_z(particle_symmetry, spin_symmetry; slave_fermion)]
                # Hermiticity
                for s in Svec
                    @test s' ≈ s
                end
                # operators should be normalized
                S = 1 / 2
                @test sum(tr(Svec[i]^2) for i in 1:3) / (2S + 1) ≈ S * (S + 1)
                # test S_plus and S_min
                @test tJ.S_plusmin(particle_symmetry, spin_symmetry; slave_fermion) ≈
                      tJ.S_plus(particle_symmetry, spin_symmetry; slave_fermion) ⊗
                      tJ.S_min(particle_symmetry, spin_symmetry; slave_fermion)
                # commutation relations
                for i in 1:3, j in 1:3
                    @test Svec[i] * Svec[j] - Svec[j] * Svec[i] ≈
                          sum(im * ε[i, j, k] * Svec[k] for k in 1:3)
                end
            end
        else
            @test_broken tJ.e_plusmin(particle_symmetry, spin_symmetry; slave_fermion)
            @test_broken tJ.e_minplus(particle_symmetry, spin_symmetry; slave_fermion)
        end
    end
end

function hamiltonian(particle_symmetry, spin_symmetry; t, J, mu, L, slave_fermion)
    num = tJ.e_number(particle_symmetry, spin_symmetry; slave_fermion)
    hop_heis = (-t) * (tJ.e_plusmin(particle_symmetry, spin_symmetry; slave_fermion) +
                       tJ.e_minplus(particle_symmetry, spin_symmetry; slave_fermion)) +
               J *
               (tJ.S_exchange(particle_symmetry, spin_symmetry; slave_fermion) -
                (1 / 4) * (num ⊗ num))
    chemical_potential = (-mu) * num
    I = id(tJ.tj_space(particle_symmetry, spin_symmetry; slave_fermion))
    H = sum(1:(L - 1)) do i
        return reduce(⊗, insert!(collect(Any, fill(I, L - 2)), i, hop_heis))
    end + sum(1:L) do i
          return reduce(⊗, insert!(collect(Any, fill(I, L - 1)), i, chemical_potential))
          end
    return H
end

@testset "spectrum" begin
    L = 4
    t = randn()
    J = randn()
    mu = randn()

    for slave_fermion in (false, true)
        H_triv = hamiltonian(Trivial, Trivial; t, J, mu, L, slave_fermion)
        vals_triv = mapreduce(vcat, eigvals(H_triv)) do (c, v)
            return repeat(real.(v), dim(c))
        end
        sort!(vals_triv)

        for (particle_symmetry, spin_symmetry) in implemented_symmetries
            if (particle_symmetry, spin_symmetry) == (Trivial, Trivial)
                continue
            end
            H_symm = hamiltonian(particle_symmetry, spin_symmetry; t, J, mu, L,
                                 slave_fermion)
            vals_symm = mapreduce(vcat, eigvals(H_symm)) do (c, v)
                return repeat(real.(v), dim(c))
            end
            sort!(vals_symm)
            @test vals_triv ≈ vals_symm
        end
    end
end
