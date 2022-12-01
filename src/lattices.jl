"""
    abstract type AbstractLattice end

Abstract supertype of all lattices.
"""
abstract type AbstractLattice end

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

#####################
## Infinite Chains ##
#####################
"""
    InfiniteChain(L::Integer=1)

A one dimensional infinite lattice with a unit cell containing `L` sites.
"""
struct InfiniteChain <: AbstractLattice
    L::Int
    function InfiniteChain(L::Integer=1)
        return L > 0 ? new(L) : error("period should be positive ($L)")
    end
end

vertices(chain::InfiniteChain) = 1:(chain.L)
nearest_neighbours(chain::InfiniteChain) = zip(1:(chain.L), 2:(chain.L + 1))
function bipartition(chain::InfiniteChain)
    iseven(chain.L) || throw(ArgumentError("given lattice is not bipartite."))
    return 1:2:(chain.L), 2:2:(chain.L)
end

###################
## Finite chains ##
###################
"""
    FiniteChain(length::Integer=1)

A one-dimensional lattice of length `L`
"""
struct FiniteChain <: AbstractLattice
    L::Int
    function FiniteChain(L::Integer=1)
        return L > 0 ? new(L) : error("length should be positive ($L)")
    end
end

vertices(chain::FiniteChain) = 1:(chain.L)
nearest_neighbours(chain::FiniteChain) = zip(1:(chain.L - 1), 2:(chain.L))

########################
## Infinite cylinders ##
########################
"""
    InfiniteCylinder(L::Integer, N::Integer)

Represents an infinite cylinder with `L` sites per rung and `N` sites per unit cell. 
"""
struct InfiniteCylinder <: AbstractLattice
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
struct InfiniteHelix <: AbstractLattice
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
struct InfiniteLadder <: AbstractLattice
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
    verts = (LatticePoint((i, j), ladder) for i in 1:(ladder.L - 1), j in 1:(ladder.N ÷ ladder.L))
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
## SnakePattern ##
##################
"""
    SnakePattern(lattice, pattern)

Represents a given lattice with a linear order that is provided by `pattern`.
"""
struct SnakePattern{G,F} <: AbstractLattice
    lattice::G
    pattern::F
end

SnakePattern(lattice) = SnakePattern(lattice, identity)

function linearize_index(snake::SnakePattern, i...)
    return snake.pattern(linearize_index(snake.lattice, i...))
end

vertices(snake::SnakePattern) = vertices(snake.lattice)
nearest_neighbours(snake::SnakePattern) = vertices(snake.lattice)
bipartition(snake::SnakePattern) = bipartition(snake.lattice)


"""
    backandforth_pattern(cylinder)

pattern that alternates directions between different rungs of a cylinder
"""
function backandforth_pattern(cylinder::InfiniteCylinder)
    iseven(cylinder.period) || error("backandforth only defined for even period")

    L = cylinder.circumference
    P = cylinder.period

    inds = Iterators.flatten((1:L, reverse((L + 1):(2L))) .+ (L * (i - 1)) for i in 1:2:P)

    return pattern(i::Integer) = inds[i]
end

"""
    frontandback_pattern(cylinder)

pattern that alternates between a site on the first half of a rung and a site on the second
half of a rung.
"""
function frontandback_pattern(cylinder::InfiniteCylinder)
    L = cylinder.circumference
    P = cylinder.period

    if iseven(L)
        edge = L / 2
        rung = Iterators.flatten(zip(1:edge, (edge + 1):L))
    else
        edge = (L + 1) / 2
        rung = Iterators.flatten((Iterators.flatten(zip(1:edge, (edge + 1):L)), edge))
    end

    inds = Iterators.flatten((rung .+ (L * (i - 1)) for i in 1:P))

    return pattern(i::Integer) = inds[i]
end

##################
## LatticePoint ##
##################
"""
    LatticePoint{N,G}
    
represents an `N`-dimensional point on a `G` lattice.
"""
struct LatticePoint{N,G<:AbstractLattice}
    coordinates::NTuple{N,Int}
    lattice::G
end

function LatticePoint(inds::NTuple{2,Int}, lattice::InfiniteCylinder)
    modded_inds = (mod1(inds[1], lattice.circumference), inds[2])
    return LatticePoint{2,InfiniteCylinder}(modded_inds, lattice)
end

linearize_index(p::LatticePoint) = linearize_index(p.lattice, p.coordinates...)
Base.to_index(p::LatticePoint) = linearize_index(p)

function Base.:+(i::LatticePoint{N,G}, j::LatticePoint{N,G}) where {N,G}
    i.lattice == j.lattice || error("can only add points of the same lattice")
    return LatticePoint{N,G}(i.coordinates .+ j.coordinates, i.lattice)
end
function Base.:+(i::LatticePoint{N,G}, j::NTuple{N,Int}) where {N,G}
    return i + LatticePoint(j, i.lattice)
end
Base.:+(i::NTuple{N,Int}, j::LatticePoint{N,G}) where {N,G} = j + i
