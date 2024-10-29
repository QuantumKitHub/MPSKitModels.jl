module MPSKitModels

using TensorKit, MPSKit, BlockTensorKit
using MacroTools: @capture, postwalk
using MPSKit: @plansor, _lastspace, _firstspace
using TensorOperations
using TupleTools

using LinearAlgebra: LinearAlgebra

export AbstractLattice, AbstractFiniteLattice, AbstractInfiniteLattice
export InfiniteChain, FiniteChain
export InfiniteCylinder, InfiniteHelix, InfiniteStrip, InfiniteLadder
export HoneycombXC, HoneycombYC
export LatticePoint, linearize_index
export vertices, nearest_neighbours, next_nearest_neighbours, bipartition
export SnakePattern, frontandback_pattern, backandforth_pattern

export LocalOperator, SumOfLocalOperators
export @mpoham

export spinmatrices, nonsym_spintensors, nonsym_bosonictensors, weyl_heisenberg_matrices

export S_x, S_y, S_z, S_plus, S_min
export S_xx, S_yy, S_zz, S_plusmin, S_minplus, S_exchange
export Sˣ, Sʸ, Sᶻ, S⁺, S⁻, Sˣˣ, Sʸʸ, Sᶻᶻ, S⁺⁻, S⁻⁺, SS
export σˣ, σʸ, σᶻ, σ⁺, σ⁻, σˣˣ, σʸʸ, σᶻᶻ, σ⁺⁻, σ⁻⁺, σσ
export potts_X, potts_Z, potts_field, potts_ZZ

export a_plus, a_min, a_plusmin, a_minplus, a_number
export a⁺, a⁻

export c_plus, c_min, c_plusplus, c_minmin, c_plusmin, c_minplus, c_number
export c⁺, c⁻, c⁺⁺, c⁻⁻, c⁺⁻, c⁻⁺
export e_plus, e_min, e_plusplus, e_minmin, e_plusmin, e_minplus
export e_number, e_number_up, e_number_down, e_number_updown
export e⁺⁺, e⁻⁻, e⁺⁻, e⁻⁺
export TJOperators

export transverse_field_ising
export kitaev_model
export quantum_potts
export heisenberg_XXX, heisenberg_XXZ, heisenberg_XYZ
export bilinear_biquadratic_model
export hubbard_model, bose_hubbard_model
export tj_model
export quantum_chemistry_hamiltonian

export classical_ising
export sixvertex
export hard_hexagon
export qstate_clock

include("utility.jl")

include("lattices/lattices.jl")
include("lattices/latticepoints.jl")
include("lattices/chains.jl")
include("lattices/squarelattice.jl")
include("lattices/triangularlattice.jl")
include("lattices/snakepattern.jl")

include("operators/localoperators.jl")
include("operators/mpoham.jl")

include("operators/spinoperators.jl")
include("operators/fermionoperators.jl")
include("operators/hubbardoperators.jl")
using .HubbardOperators
# TJOperators share operator names with HubbardOperators
# and is only imported to avoid name conflicts
include("operators/tjoperators.jl")
import .TJOperators
include("operators/bosonoperators.jl")

include("models/hamiltonians.jl")
include("models/quantum_chemistry.jl")
include("models/transfermatrices.jl")

# disable precompilation until MPOHamiltonian is type stable
# otherwise this takes annoyingly long
# include("precompile.jl")

end
