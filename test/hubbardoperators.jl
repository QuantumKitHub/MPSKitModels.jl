using Test
using TensorKit
using MPSKitModels.HubbardOperators
using LinearAlgebra: eigvals

implemented_symmetries = [(Trivial, Trivial), (U1Irrep, U1Irrep), (U1Irrep, SU2Irrep)]
@testset "basic properties" begin
    for particle_symmetry in (Trivial, U1Irrep, SU2Irrep),
        spin_symmetry in (Trivial, U1Irrep, SU2Irrep)

        if (particle_symmetry, spin_symmetry) in implemented_symmetries
            # test hermiticity
            @test e_plusmin(particle_symmetry, spin_symmetry)' ≈
                  e_minplus(particle_symmetry, spin_symmetry)
            if spin_symmetry !== SU2Irrep
                @test e_plusmin_down(particle_symmetry, spin_symmetry)' ≈
                      e_minplus_down(particle_symmetry, spin_symmetry)
                @test e_plusmin_up(particle_symmetry, spin_symmetry)' ≈
                      e_minplus_up(particle_symmetry, spin_symmetry)
                @test e_plusmin_down(particle_symmetry, spin_symmetry)' ≈
                      e_minplus_down(particle_symmetry, spin_symmetry)
                @test e_plusmin_up(particle_symmetry, spin_symmetry)' ≈
                      e_minplus_up(particle_symmetry, spin_symmetry)
            end

            # test number operator
            if spin_symmetry !== SU2Irrep
                @test e_number(particle_symmetry, spin_symmetry) ≈
                      e_number_up(particle_symmetry, spin_symmetry) +
                      e_number_down(particle_symmetry, spin_symmetry)
                @test e_number_updown(particle_symmetry, spin_symmetry) ≈
                      e_number_up(particle_symmetry, spin_symmetry) *
                      e_number_down(particle_symmetry, spin_symmetry) ≈
                      e_number_down(particle_symmetry, spin_symmetry) *
                      e_number_up(particle_symmetry, spin_symmetry)
            end
        else
            @test_broken e_plusmin(particle_symmetry, spin_symmetry)
            @test_broken e_minplus(particle_symmetry, spin_symmetry)
        end
    end
end

function hubbard_hamiltonian(particle_symmetry, spin_symmetry; t, U, mu, L)
    hopping = t * (e_plusmin(particle_symmetry, spin_symmetry) +
                   e_minplus(particle_symmetry, spin_symmetry))
    interaction = U * e_number_updown(particle_symmetry, spin_symmetry)
    chemical_potential = mu * e_number(particle_symmetry, spin_symmetry)
    I = id(hubbard_space(particle_symmetry, spin_symmetry))
    H = sum(1:(L - 1)) do i
            return reduce(⊗, insert!(collect(Any, fill(I, L - 2)), i, hopping))
        end +
        sum(1:L) do i
            return reduce(⊗, insert!(collect(Any, fill(I, L - 1)), i, interaction))
        end +
        sum(1:L) do i
            return reduce(⊗, insert!(collect(Any, fill(I, L - 1)), i, chemical_potential))
        end
    return H
end

@testset "spectrum" begin
    L = 4
    t = randn()
    U = randn()
    mu = randn()

    H_triv = hubbard_hamiltonian(Trivial, Trivial; t, U, mu, L)
    vals_triv = mapreduce(vcat, eigvals(H_triv)) do (c, v)
        return repeat(real.(v), dim(c))
    end
    sort!(vals_triv)

    for (particle_symmetry, spin_symmetry) in implemented_symmetries
        if (particle_symmetry, spin_symmetry) == (Trivial, Trivial)
            continue
        end
        H_symm = hubbard_hamiltonian(particle_symmetry, spin_symmetry; t, U, mu, L)
        vals_symm = mapreduce(vcat, eigvals(H_symm)) do (c, v)
            return repeat(real.(v), dim(c))
        end
        sort!(vals_symm)
        @test vals_triv ≈ vals_symm
    end
end
