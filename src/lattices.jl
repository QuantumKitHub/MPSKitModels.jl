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

linearize_index(::AbstractLattice, i::Int) = i

#####################
## Infinite Chains ##
#####################
"""
    InfiniteChain(L::Integer=1)

Represents a one dimensional infinite lattice with a unit cell containing `L` sites.
"""
struct InfiniteChain <: AbstractLattice
    L::Int
    function InfiniteChain(L::Integer=1)
        return L > 0 ? new(L) : error("period should be positive ($L)")
    end
end

vertices(chain::InfiniteChain) = 1:chain.L
nearest_neighbours(chain::InfiniteChain) = zip(1:chain.L, 2:chain.L+1)

###################
## Finite chains ##
###################
"""
    FiniteChain(length::Integer=1)

Represents a one-dimensional lattice of length `L`
"""
struct FiniteChain <: AbstractLattice
    L::Int
    function FiniteChain(L::Integer=1)
        return L > 0 ? new(L) : error("length should be positive ($L)")
    end
end

vertices(chain::FiniteChain) = 1:chain.L
nearest_neighbours(chain::FiniteChain) = zip(1:chain.L-1, 2:chain.L)

########################
## Infinite cylinders ##
########################

"""
    InfiniteCylinder(L::Integer, N::Integer)

Represents an infinite cylinder with `L` sites per rung and `N` rungs per unit cell. 
"""
struct InfiniteCylinder <: AbstractLattice
    circumference::Int
    period::Int
    function InfiniteCylinder(circumference, period=1)
        period > 0 || error("period should be positive")
        return new(circumference, period)
    end
end

function linearize_index(cylinder::InfiniteCylinder, i::Int, j::Int)
    return mod1(i, cylinder.circumference) + cylinder.circumference * (j - 1)
end

function vertices(cylinder::InfiniteCylinder)
    return (LatticePoint((i, j), cylinder) for i in 1:cylinder.circumference, j in 1:cylinder.period)
end

function radial_neighbours(cylinder::InfiniteCylinder)
    radial_shift = LatticePoint((1, 0), cylinder)
    return zip(vertices(cylinder), vertices(cylinder) .+ radial_shift)
end

function axial_neighbours(cylinder::InfiniteCylinder)
    axial_shift = LatticePoint((0, 1), cylinder)
    return zip(vertices(cylinder), vertices(cylinder) .+ axial_shift)
end

function nearest_neighbours(cylinder::InfiniteCylinder)
    return Iterators.flatten((radial_neighbours(cylinder), axial_neighbours(cylinder)))
end

######################
## Infinite Spirals ##
######################


######################
## Infinite ladders ##
######################


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

"""
    backandforth_pattern(cylinder)

pattern that alternates directions between different rungs of a cylinder
"""
function backandforth_pattern(cylinder::InfiniteCylinder)
    iseven(cylinder.period) || error("backandforth only defined for even period")

    L = cylinder.circumference
    P = cylinder.period

    inds = Iterators.flatten((1:L, reverse(L+1:2L)) .+ (L * (i - 1)) for i in 1:2:P)

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
        rung = Iterators.flatten(zip(1:edge, edge+1:L))
    else
        edge = (L + 1) / 2
        rung = Iterators.flatten((Iterators.flatten(zip(1:edge, edge+1:L)), edge))
    end

    inds = Iterators.flatten((rung .+ (L * (i - 1)) for i in 1:P))

    return pattern(i::Integer) = inds[i]
end

##################
## LatticePoint ##
##################
struct LatticePoint{N,G<:AbstractLattice}
    coordinates::NTuple{N,Int}
    lattice::G
end

function LatticePoint(inds::NTuple{2,Int}, lattice::InfiniteCylinder)
    modded_inds = (mod1(inds[1], lattice.circumference), inds[2])
    return LatticePoint{2,InfiniteCylinder}(modded_inds, lattice)
end

linearize_index(p::LatticePoint) = linearize_index(p.lattice, p.coordinates...)

function Base.:+(i::LatticePoint{N,G}, j::LatticePoint{N,G}) where {N,G}
    i.lattice == j.lattice || error("can only add points of the same lattice")
    return LatticePoint{N,G}(i.coordinates .+ j.coordinates, i.lattice)
end
function Base.:+(i::LatticePoint{N,G}, j::NTuple{N,Int}) where {N,G}
    return i + LatticePoint(j, i.lattice)
end
Base.:+(i::NTuple{N,Int}, j::LatticePoint{N,G}) where {N,G} = j + i
