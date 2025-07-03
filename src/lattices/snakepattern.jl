"""
    SnakePattern(lattice, pattern)

Represents a given lattice with a linear order that is provided by `pattern`.
"""
struct SnakePattern{N,G<:AbstractLattice{N},F} <: AbstractLattice{N}
    lattice::G
    pattern::F
end

SnakePattern(lattice) = SnakePattern(lattice, identity)

Base.axes(lattice::SnakePattern) = axes(lattice.lattice)
Base.isfinite(::Type{SnakePattern{N,G}}) where {N,G} = isfinite(G)

function linearize_index(snake::SnakePattern, i...)
    return snake.pattern(linearize_index(snake.lattice, i...))
end

vertices(snake::SnakePattern) = vertices(snake.lattice)
nearest_neighbours(snake::SnakePattern) = nearest_neighbours(snake.lattice)
bipartition(snake::SnakePattern) = bipartition(snake.lattice)

"""
    backandforth_pattern(cylinder)

pattern that alternates directions between different rungs of a cylinder
"""
function backandforth_pattern(cylinder::InfiniteCylinder)
    L = cylinder.L
    N = cylinder.N
    iseven(cylinder.N) || error("backandforth only defined for even period")
    inds = Iterators.flatten((1:L, reverse((L + 1):(2L))) .+ (L * (i - 1)) for i in 1:2:N)

    return pattern(i::Integer) = inds[i]
end

"""
    frontandback_pattern(cylinder)

pattern that alternates between a site on the first half of a rung and a site on the second
half of a rung.
"""
function frontandback_pattern(cylinder::InfiniteCylinder)
    L = cylinder.L
    N = cylinder.N

    if iseven(L)
        edge = L / 2
        rung = Iterators.flatten(zip(1:edge, (edge + 1):L))
    else
        edge = (L + 1) / 2
        rung = Iterators.flatten((Iterators.flatten(zip(1:edge, (edge + 1):L)), edge))
    end

    inds = Iterators.flatten((rung .+ (L * (i - 1)) for i in 1:N))

    return pattern(i::Integer) = inds[i]
end
