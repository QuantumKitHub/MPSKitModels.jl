"""
    InfiniteChain(L::Integer=1)

A one dimensional infinite lattice with a unit cell containing `L` sites.
"""
struct InfiniteChain <: AbstractLattice{1}
    L::Int
    function InfiniteChain(L::Integer=1)
        return L > 0 ? new(L) : error("period should be positive ($L)")
    end
end

"""
    FiniteChain(length::Integer=1)

A one-dimensional lattice of length `L`
"""
struct FiniteChain <: AbstractLattice{1}
    L::Int
    function FiniteChain(L::Integer=1)
        return L > 0 ? new(L) : error("length should be positive ($L)")
    end
end

const Chain = Union{InfiniteChain,FiniteChain}

vertices(chain::Chain) = LatticePoint.(1:(chain.L), Ref(chain))
nearest_neighbours(chain::InfiniteChain) = map(v -> v => v + 1, vertices(chain))
nearest_neighbours(chain::FiniteChain) = map(v -> v => v + 1, Base.front(vertices(chain)))
next_nearest_neighbours(chain::InfiniteChain) = map(v -> v => v + 2, vertices(chain))
next_nearest_neighbours(chain::FiniteChain) = map(v -> v => v + 2, Base.front(Base.front(vertices(chain))))

function bipartition(chain::Chain)
    A = map(i -> LatticePoint(i, chain), 1:2:(chain.L))
    B = map(i -> LatticePoint(i, chain), 2:2:(chain.L))
    return A, B
end

linearize_index(chain::InfiniteChain, i::Int) = i
linearize_index(chain::Chain, i::NTuple{1,Int}) = linearize_index(chain, i...)
function linearize_index(chain::FiniteChain, i::Int)
    0 < i <= chain.L || throw(BoundsError("lattice point out of bounds"))
    return i
end

LinearAlgebra.norm(p::LatticePoint{1,<:Union{FiniteChain,InfiniteChain}}) = 
    abs(p.coordinates[1])