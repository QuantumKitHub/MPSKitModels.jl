using MPSKitModels
using TensorKit
using TensorOperations
using Test

cutoff = 3
elt = ComplexF64

using MPSKitModels: contract_twosite

@testset "$symmetry" for symmetry in (ℤ{1}, U₁)
    a_plusmin = contract_twosite(a_plus(cutoff, elt, U₁; side=:L), 
                                 a_min(cutoff, elt, U₁; side=:R))
    a_minplus = contract_twosite(a_min(cutoff, elt, U₁; side=:L), 
                                 a_plus(cutoff, elt, U₁; side=:R))
    @test a_plusmin ≈ adjoint(a_minplus)
end
