using Test
using MPSKitModels

@testset "lattices" verbose = true begin
    include("lattices.jl")
end

@testset "Spin operators" begin
    include("spinoperators.jl")
end

@testset "Boson operators" begin
    include("bosonoperators.jl")
end

@testset "fermion operators" begin
    include("fermionoperators.jl")
end

@testset "Hubbard operators" begin
    include("hubbardoperators.jl")
end

@testset "t-J operators" begin
    include("tjoperators.jl")
end

@testset "mpoham" begin
    include("mpoham.jl")
end

@testset "transverse field ising model" begin
    include("tfim.jl")
end

@testset "heisenberg model" begin
    include("heisenberg.jl")
end

@testset "bose-hubbard model" begin
    include("bose_hubbard.jl")
end

@testset "quantum potts model" begin
    include("quantum_potts.jl")
end

@testset "classical ising model" begin
    include("classical_ising.jl")
end

@testset "sixvertex model" begin
    include("sixvertex.jl")
end
