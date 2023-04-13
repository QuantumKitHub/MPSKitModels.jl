"""
    AbstractLattice{N}

Abstract supertype of all lattices, which are mapped to `N`-dimensional integer grids.
"""
abstract type AbstractLattice{N} end

"""
    vertices(lattice)

construct an iterator over all lattice points.
"""
function vertices end

"""
    nearest_neighbours(lattice)

construct an iterator over all pairs of nearest neighbours.
"""
function nearest_neighbours end

"""
    bipartition(lattice)

construct two iterators over the vertices of the bipartition of a given lattice.
"""
function bipartition end

"""
    linearize_index(lattice, indices...)

convert a given set of indices into a linear index.
"""
linearize_index(::AbstractLattice, i::Int) = i


Base.length(L::AbstractLattice) = length(vertices(L))
Base.iterate(L::AbstractLattice) = iterate(vertices(L))



########################
## Infinite cylinders ##
########################
"""
    InfiniteCylinder(L::Integer, N::Integer)

Represents an infinite cylinder with `L` sites per rung and `N` sites per unit cell. 
"""
struct InfiniteCylinder <: AbstractLattice{2}
    L::Int
    N::Int
    function InfiniteCylinder(L::Integer, N::Integer=L)
        N > 0 || error("period should be positive")
        mod(N, L) == 0 ||
            error("period should be a multiple of circumference")
        return new(L, N)
    end
end

function linearize_index(cylinder::InfiniteCylinder, i::Int, j::Int)
    return mod1(i, cylinder.L) + cylinder.L * (j - 1)
end

function vertices(cylinder::InfiniteCylinder)
    return (LatticePoint((i, j), cylinder) for i in 1:(cylinder.L),
                                               j in 1:(cylinder.N ÷ cylinder.L))
end

function radial_neighbours(cylinder::InfiniteCylinder)
    radial_shift = LatticePoint((1, 0), cylinder)
    return zip(vertices(cylinder), vertices(cylinder) .+ (radial_shift,))
end

function axial_neighbours(cylinder::InfiniteCylinder)
    axial_shift = LatticePoint((0, 1), cylinder)
    return zip(vertices(cylinder), vertices(cylinder) .+ (axial_shift,))
end

function nearest_neighbours(cylinder::InfiniteCylinder)
    return Iterators.flatten((radial_neighbours(cylinder), axial_neighbours(cylinder)))
end

######################
## Infinite Helices ##
######################
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

function linearize_index(helix::InfiniteHelix, i::Int, j::Int)
    return mod1(i, helix.L) + helix.L * (j + (i - 1) ÷ helix.L - 1)
end

function vertices(helix::InfiniteHelix)
    return (LatticePoint((i, 1), helix) for i in 1:(helix.N))
end

function radial_neighbours(helix::InfiniteHelix)
    radial_shift = LatticePoint((1, 0), helix)
    return zip(vertices(helix), vertices(helix) .+ (radial_shift,))
end

function axial_neighbours(helix::InfiniteHelix)
    axial_shift = LatticePoint((0, 1), helix)
    return zip(vertices(helix), vertices(helix) .+ (axial_shift,))
end

function nearest_neighbours(helix::InfiniteHelix)
    return Iterators.flatten((radial_neighbours(helix), axial_neighbours(helix)))
end

######################
## Infinite ladders ##
######################
"""
    InfiniteLadder(L::Integer, N::Integer)

An infinite ladder with `L` sites per rung, `N` sites per unit cell.
"""
struct InfiniteLadder <: AbstractLattice{2}
    L::Int
    N::Int
    function InfiniteLadder(L::Integer, N::Integer=L)
        mod(N, L) == 0 ||
            error("period should be a multiple of circumference")
        return new(L, N)
    end
end

function linearize_index(ladder::InfiniteLadder, i::Int, j::Int)
    @assert i <= ladder.L "lattice point out of range"
    return i + (j - 1) * ladder.L
end

function vertices(ladder::InfiniteLadder)
    return (LatticePoint((i, j), ladder) for i in 1:(ladder.L),
                                             j in 1:(ladder.N ÷ ladder.L))
end

function radial_neighbours(ladder::InfiniteLadder)
    radial_shift = LatticePoint((1, 0), ladder)
    verts = (LatticePoint((i, j), ladder) for i in 1:(ladder.L - 1),
                                              j in 1:(ladder.N ÷ ladder.L))
    return zip(verts, verts .+ radial_shift)
end

function axial_neighbours(ladder::InfiniteLadder)
    axial_shift = LatticePoint((0, 1), ladder)
    return zip(vertices(ladder), vertices(ladder) .+ axial_shift)
end

function nearest_neighbours(ladder::InfiniteLadder)
    return Iterators.flatten((radial_neighbours(ladder), axial_neighbours(ladder)))
end



##################
## LatticePoint ##
##################
