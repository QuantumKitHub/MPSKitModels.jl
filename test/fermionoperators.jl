using MPSKitModels
using TensorKit
using TensorOperations
using Test
using LinearAlgebra: tr

# anticommutation relations
# {cᵢ†, cⱼ†} = 0 = {cᵢ, cⱼ}

@test cc() ≈ -permute(cc(), (2, 1), (4, 3))
@test cdagcdag() ≈ -permute(cdagcdag(), (2, 1), (4, 3))
@test ccdag() ≈ -permute(cdagc(), (2, 1), (4, 3))

@tensor begin
    term1[a; b] := ccdag()[a, i; i, b]
    term2[a; b] := cdagc()[a, i; i, b]
end

@test term1 + term2 ≈ id(domain(term1))

@test term2 ≈ number()