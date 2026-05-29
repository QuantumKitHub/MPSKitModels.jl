using Test: @testset

using MPSKitModels: quantum_chemistry_hamiltonian

## Setup
# H2 molecule with sto-3g basis and molecular orbitals from Hartree-Fock

ecore = 0.7178535240637794
hpq = [-1.2550254253591244 0.0; 0.0 -0.4732763494710688]
hpqrs = zeros((2, 2, 2, 2))
hpqrs[1, 1, 1, 1] = 0.6752967689354994
hpqrs[2, 2, 1, 1] = 0.6642044392432876
hpqrs[2, 1, 2, 1] = 0.18105207136899099
hpqrs[1, 2, 2, 1] = 0.18105207136899099
hpqrs[2, 1, 1, 2] = 0.18105207136899099
hpqrs[1, 2, 1, 2] = 0.18105207136899099
hpqrs[1, 1, 2, 2] = 0.6642044392432875
hpqrs[2, 2, 2, 2] = 0.6981738857839888

@testset "Quantum chemistry Hamiltonian" begin
    mpo = quantum_chemistry_hamiltonian(ecore, hpq, hpqrs, Float64)
end
