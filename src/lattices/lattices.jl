"""
    AbstractLattice{N}

Abstract supertype of all lattices, which are mapped to `N`-dimensional integer grids.
"""
abstract type AbstractLattice{N} end

"""
    vertices(lattice::AbstractLattice)

construct an iterator over all lattice points.
"""
function vertices end

"""
    nearest_neighbours(lattice::AbstractLattice)

construct an iterator over all pairs of nearest neighbours.
"""
function nearest_neighbours end

"""
    next_nearest_neighbours(lattice::AbstractLattice)

construct an iterator over all pairs of next-nearest neighbours.
"""
function next_nearest_neighbours end

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
