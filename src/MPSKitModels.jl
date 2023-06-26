module MPSKitModels

using TensorKit, MPSKit
using LinearAlgebra: Diagonal, diag
using MacroTools: @capture, postwalk
using MPSKit: @plansor, _lastspace, _firstspace
using TensorOperations

import LinearAlgebra

export AbstractLattice
export InfiniteChain, FiniteChain
export InfiniteCylinder, InfiniteHelix, InfiniteStrip
export HoneycombXC, HoneycombYC
export LatticePoint, linearize_index
export vertices, nearest_neighbours, bipartition
export SnakePattern, frontandback_pattern, backandforth_pattern

export spinmatrices, nonsym_spintensors, nonsym_bosonictensors

export sigma_x, sigma_y, sigma_z, sigma_plus, sigma_min
export sigma_xx, sigma_yy, sigma_zz, sigma_plusmin, sigma_minplus, sigma_exchange
export σˣ, σʸ, σᶻ, σ⁺, σ⁻, σˣˣ, σʸʸ, σᶻᶻ, σ⁺⁻, σ⁻⁺, σσ

export a_plus, a_min, a_number
export a⁺, a⁻

export c_plus, c_min, c_plusplus, c_minmin, c_plusmin, c_minplus, c_number
export c⁺, c⁻, c⁺⁺, c⁻⁻, c⁺⁻, c⁻⁺ 
export e_plus, e_min, e_plusplus, e_minmin, e_plusmin, e_minplus
export e_number, e_number_up, e_number_down, e_number_updown
export e⁺, e⁻, e⁺⁺, e⁻⁻, e⁺⁻, e⁻⁺





include("utility.jl")


include("lattices/lattices.jl")
include("lattices/latticepoints.jl")
include("lattices/chains.jl")
include("lattices/squarelattice.jl")
include("lattices/triangularlattice.jl")
include("lattices/snakepattern.jl")

export LocalOperator, SumOfLocalOperators
include("operators/localoperators.jl")

const ℤ₁ = ℤ{1}

include("operators/spinoperators.jl")

include("operators/fermionoperators.jl")

include("operators/bosonoperators.jl")

export @mpoham
include("mpoham.jl")

export transverse_field_ising, free_fermion_ising
export xxx, xxz, xyz
export bilinear_biquadratic_heisenberg
export hubbard_model, bose_hubbard_model
include("models/hamiltonians.jl")

export classical_ising
export sixvertex
export hard_hexagon
export qstate_clock
include("models/transfermatrices.jl")



export nonsym_xxz_ladder_finite, nonsym_xxz_ladder_infinite, su2_xxx_ladder
include("xxz.jl")

export su2u1_grossneveu, su2u1_orderpars, su2su2_grossneveu, su2su2_orderpars
include("grossneveu.jl")

export nonsym_qed_qlm_ham, qed_qlm_G2, u1_qed_ham
include("qed_qlm.jl")

export U1_strip_harper_hofstadter
include("hofstadter.jl")

export quantum_chemistry_hamiltonian
include("quantum_chemistry.jl")

include("deprecate.jl")

end
