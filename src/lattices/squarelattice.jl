"""
    InfiniteStrip(L::Int, N::Int)

An infinite strip with `L` sites per rung and `N` sites per unit cell.
"""
struct InfiniteStrip <: AbstractLattice{2}
    L::Int
    N::Int
    function InfiniteStrip(L::Integer, N::Integer=L)
        N > 0 || throw(ArgumentError("period should be positive"))
        mod(N, L) == 0 ||
            throw(ArgumentError("period should be a multiple of circumference"))
        return new(L, N)
    end
end
InfiniteLadder(N::Integer) = InfiniteStrip(2, N)
Base.axes(strip::InfiniteStrip) = (1:(strip.L), (-typemax(Int)):typemax(Int))

"""
    InfiniteCylinder(L::Int, N::Int)

An infinite cylinder with `L` sites per rung and `N` sites per unit cell. 
"""
struct InfiniteCylinder <: AbstractLattice{2}
    L::Int
    N::Int
    function InfiniteCylinder(L::Integer, N::Integer=L)
        N > 0 || throw(ArgumentError("period should be positive"))
        mod(N, L) == 0 ||
            throw(ArgumentError("period should be a multiple of circumference"))
        return new(L, N)
    end
end

Base.axes(::InfiniteCylinder) = (-typemax(Int):typemax(Int), -typemax(Int):typemax(Int))

"""
    InfiniteHelix(L::Integer, N::Integer)

An infinite helix with `L` sites per rung and `N` sites per unit cell.
"""
struct InfiniteHelix <: AbstractLattice{2}
    L::Int
    N::Int
    function InfiniteHelix(L::Integer, N::Integer=1)
        N > 0 || error("period should be positive")
        return new(L, N)
    end
end

function Base.axes(::InfiniteHelix)
    return ((-typemax(Int)):typemax(Int), (-typemax(Int)):typemax(Int))
end

############################################################################################

function linearize_index(lattice::InfiniteStrip, i::Int, j::Int)
    @assert i <= lattice.L "lattice point out of bounds"
    return mod1(i, lattice.L) + lattice.L * (j - 1)
end
function linearize_index(lattice::InfiniteCylinder, i::Int, j::Int)
    return mod1(i, lattice.L) + lattice.L * (j - 1)
end
function linearize_index(helix::InfiniteHelix, i::Int, j::Int)
    return mod1(i, helix.L) + helix.L * (j + (i - 1) ÷ helix.L - 1)
end

function vertices(lattice::Union{InfiniteStrip, InfiniteCylinder})
    return (LatticePoint((i, j), lattice) for i in 1:(lattice.L), j in 1:(lattice.N ÷ lattice.L))
end
vertices(lattice::InfiniteHelix) = (LatticePoint((i,1), lattice) for i in 1:(lattice.N))

function nearest_neighbours(lattice::Union{InfiniteStrip, InfiniteCylinder, InfiniteHelix})
    V = vertices(lattice)
    neighbours = Pair{eltype(V), eltype(V)}[]
    for v in V
        push!(neighbours, v => v + (0, 1))
        if v.coordinates[1] < lattice.L || lattice isa InfiniteCylinder || lattice isa InfiniteHelix
            push!(neighbours, v => v + (1, 0))
        end
    end
    return neighbours
end

function next_nearest_neighbours(lattice::Union{InfiniteStrip, InfiniteCylinder, InfiniteHelix})
    V = vertices(lattice)
    neighbours = Pair{eltype(V), eltype(V)}[]
    for v in V
        if v.coordinates[1] < lattice.L || lattice isa InfiniteCylinder || lattice isa InfiniteHelix
            push!(neighbours, v => v + (1, 1))
        end
        if v.coordinates[1] > 1 || lattice isa InfiniteCylinder || lattice isa InfiniteHelix
            push!(neighbours, v => v + (-1, 1))
        end
    end
    return neighbours
end

LinearAlgebra.norm(p::LatticePoint{2,InfiniteStrip}) = LinearAlgebra.norm(p.coordinates)
function LinearAlgebra.norm(p::LatticePoint{2,InfiniteCylinder})
    return min(sqrt(mod(p.coordinates[1], p.lattice.L)^2 + p.coordinates[2]^2), sqrt(mod(-p.coordinates[1], p.lattice.L)^2 + p.coordinates[2]^2))
end
function LinearAlgebra.norm(p::LatticePoint{2,InfiniteHelix})
    x₁ = mod(p.coordinates[1], p.lattice.L)
    y₁ = p.coordinates[2] + (p.coordinates[1] ÷ p.lattice.L)
    x₂ = mod(-p.coordinates[1], p.lattice.L)
    y₂ = p.coordinates[2] - (p.coordinates[1] ÷ p.lattice.L)
    return min(sqrt(x₁^2 + y₁^2), sqrt(x₂^2 + y₂^2))
end