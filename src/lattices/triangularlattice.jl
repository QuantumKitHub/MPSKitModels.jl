"""
    HoneycombYC(L::Integer, N::Integer=L)

A honeycomb lattice on an infinite cylinder with `L` sites per rung and `N` sites per unit cell.
The y-axis is aligned along an edge of the hexagons, and the circumference is ``3L/4``.
"""
struct HoneycombYC <: AbstractLattice{2}
    L::Int
    N::Int
    function HoneycombYC(L::Integer, N::Integer = L)
        (L > 0 && N > 0) ||
            throw(ArgumentError("period and length should be strictly positive"))
        mod(L, 4) == 0 || throw(ArgumentError("period must be a multiple of 4"))
        mod(N, L) == 0 ||
            throw(ArgumentError("period should be a multiple of circumference"))
        return new(L, N)
    end
end

Base.isfinite(::Type{HoneycombYC}) = false

# TODO: do proper boundscheck
function Base.checkbounds(::Type{Bool}, lattice::HoneycombYC, inds::Vararg{Int, 2})
    return true
end

function LinearAlgebra.norm(p::LatticePoint{2, HoneycombYC})
    x = p.coordinates[1] + p.coordinates[2] * cos(2π / 6)
    y = p.coordinates[2] * sin(2π / 6)
    x₁ = mod(x, 3p.lattice.L ÷ 4)
    x₂ = mod(-x, 3p.lattice.L ÷ 4)
    return min(sqrt(x₁^2 + y^2), sqrt(x₂^2 + y^2))
end

function vertices(lattice::HoneycombYC)
    V = [LatticePoint((1, 1), lattice)]
    for i in 2:(lattice.N)
        offset = if mod(i, 4) == 2
            (0, 1)
        elseif mod(i, 4) == 3
            (1, 0)
        elseif mod(i, 4) == 0
            (1, -1)
        else
            (1, 0)
        end
        next = last(V) + offset

        if mod(i, lattice.L) == 1 # wrap around cylinder
            next = next + (-3lattice.L ÷ 4 - 1, 2)
        end

        push!(V, next)
    end
    return V
end

function linearize_index(lattice::HoneycombYC, i::Int, j::Int)
    rungoffset = (j - 1) ÷ 2
    x = i + rungoffset
    y = j
    return rungoffset * lattice.L + (x ÷ 3) + x + (1 - mod(y, 2))
end

function nearest_neighbours(lattice::HoneycombYC)
    V = vertices(lattice)

    neighbours = Pair{eltype(V), eltype(V)}[]
    for v in V
        I = linearize_index(v)
        if mod(I, 4) == 1
            push!(neighbours, (v => v + (0, 1)))
        elseif mod(I, 4) == 2
            push!(neighbours, (v => v + (1, 0)))
            push!(neighbours, (v => v + (-1, +1)))
        elseif mod(I, 4) == 3
            push!(neighbours, (v => v + (1, -1)))
            push!(neighbours, (v => v + (0, 1)))
        elseif mod(I, 4) == 0
            if mod(I, lattice.L) == 0
                push!(neighbours, (v => v - (3(lattice.L ÷ 4) - 1, 0)))
            else
                push!(neighbours, (v => v + (1, 0)))
            end
        end
    end
    return neighbours
end
