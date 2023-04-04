module MPSKitModels
using TensorKit, MPSKit
using LinearAlgebra: Diagonal, diag
using MacroTools: @capture, postwalk
using MPSKit: @plansor, _lastspace, _firstspace

include("spinoperators.jl")
export sigma_x, sigma_y, sigma_z, sigma_plus, sigma_min
export sigma_xx, sigma_yy, sigma_zz, sigma_plusmin, sigma_minplus, sigma_exchange
export σˣ, σʸ, σᶻ, σ⁺, σ⁻, σˣˣ, σʸʸ, σᶻᶻ, σ⁺⁻, σ⁻⁺, σσ

include("lattices.jl")
export AbstractLattice
export InfiniteChain, FiniteChain
export InfiniteCylinder, InfiniteHelix, InfiniteLadder
export LatticePoint, linearize_index
export vertices, nearest_neighbours, bipartition
export SnakePattern, frontandback_pattern, backandforth_pattern

include("mpoham.jl")
export @mpoham
export LocalOperator, SumOfLocalOperators

include("models.jl")
export transverse_field_ising, classical_ising
export sixvertex
export xxx, xxz, xyz

export spinmatrices, nonsym_spintensors, nonsym_bosonictensors
include("utility.jl")

export nonsym_ising_ham, nonsym_ising_mpo, z2_ising_mpo
include("ising.jl")

export nonsym_xxz_ham, su2_xxx_ham, u1_xxz_ham
export nonsym_xxz_ladder_finite, nonsym_xxz_ladder_infinite, su2_xxx_ladder
include("xxz.jl")

export su2u1_grossneveu, su2u1_orderpars, su2su2_grossneveu, su2su2_orderpars
include("grossneveu.jl")

export nonsym_qstateclock_mpo
include("qstateclock.jl")

export nonsym_qed_qlm_ham, qed_qlm_G2, u1_qed_ham
include("qed_qlm.jl")

export nonsym_sixvertex_mpo, u1_sixvertex_mpo, cu1_sixvertex_mpo
include("sixvertex.jl")

export U1_strip_harper_hofstadter
include("hofstadter.jl")

export quantum_chemistry_hamiltonian
include("quantum_chemistry.jl")

end
