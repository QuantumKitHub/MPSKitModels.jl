"""
    InfiniteKagomeYC(L::Integer, N::Integer=L)

A kagome lattice on an infinite cylinder in the YC orientation, with `L` sites per unit cell and `N` sites per unit cell period.
`L` must be a multiple of 3.

In the YC orientation, one bond type (the flat bond) runs parallel to the circumference, perpendicular to the cylinder axis.
Coordinates use a non-orthogonal basis aligned with the kagome bond directions:
the first coordinate is along the circumference (wraps with period `2L ÷ 3`), the second is along the cylinder axis.
Within each unit cell, sites are ordered as A, B, C (three sites of an upward-pointing triangle).
"""
struct InfiniteKagomeYC <: AbstractLattice{2}
    L::Int
    N::Int
    function InfiniteKagomeYC(L::Integer, N::Integer = L)
        (L > 0 && N > 0) ||
            throw(ArgumentError("L and N must be strictly positive"))
        mod(L, 3) == 0 || throw(ArgumentError("L must be a multiple of 3"))
        mod(N, L) == 0 || throw(ArgumentError("N must be a multiple of L"))
        return new(L, N)
    end
end

Base.isfinite(::Type{InfiniteKagomeYC}) = false
Base.axes(::InfiniteKagomeYC) = ((-typemax(Int)):typemax(Int), (-typemax(Int)):typemax(Int))

"""
    FiniteKagomeYC(L::Integer, N::Integer=L)

A kagome lattice on a finite cylinder in the YC orientation, with `L` sites around the circumference and `N` total sites.
`L` must be a multiple of 3 and `N` a multiple of `L`.
"""
struct FiniteKagomeYC <: AbstractLattice{2}
    L::Int
    N::Int
    function FiniteKagomeYC(L::Integer, N::Integer = L)
        (L > 0 && N > 0) ||
            throw(ArgumentError("L and N must be strictly positive"))
        mod(L, 3) == 0 || throw(ArgumentError("L must be a multiple of 3"))
        mod(N, L) == 0 || throw(ArgumentError("N must be a multiple of L"))
        return new(L, N)
    end
end

Base.isfinite(::Type{FiniteKagomeYC}) = true
function Base.axes(lattice::FiniteKagomeYC)
    return ((-typemax(Int)):typemax(Int), 1:(2 * (lattice.N ÷ lattice.L)))
end

"""
    InfiniteKagomeXC(L::Integer, N::Integer=L)

A kagome lattice on an infinite cylinder in the XC orientation, with `L` sites per unit cell
and `N` sites per unit cell period. `L` must be a multiple of 3.

In the XC orientation, one bond type (the flat bond) runs parallel to the cylinder axis,
perpendicular to the circumference. Coordinates use a non-orthogonal basis aligned with the
kagome bond directions: the first coordinate is along the circumference (wraps with period
`2L÷3`), the second is along the cylinder axis. Within each unit cell, sites are ordered as
A, C, B (three sites of an upward-pointing triangle, listed in order of increasing first
coordinate then second coordinate).
"""
struct InfiniteKagomeXC <: AbstractLattice{2}
    L::Int
    N::Int
    function InfiniteKagomeXC(L::Integer, N::Integer = L)
        (L > 0 && N > 0) ||
            throw(ArgumentError("L and N must be strictly positive"))
        mod(L, 3) == 0 || throw(ArgumentError("L must be a multiple of 3"))
        mod(N, L) == 0 || throw(ArgumentError("N must be a multiple of L"))
        return new(L, N)
    end
end

Base.isfinite(::Type{InfiniteKagomeXC}) = false
Base.axes(::InfiniteKagomeXC) = ((-typemax(Int)):typemax(Int), (-typemax(Int)):typemax(Int))

"""
    FiniteKagomeXC(L::Integer, N::Integer=L)

A kagome lattice on a finite cylinder in the XC orientation, with `L` sites around the circumference and `N` total sites.
`L` must be a multiple of 3 and `N` a multiple of `L`.
For example, `FiniteKagomeXC(6, 12)` is equivalent to:
```
 ╲     ╱ ╲             ╲── y
  1───2───7───8         ╲
   ╲ ╱     ╲ ╱           x
    3       9             
     ╲     ╱ ╲      
      4───5───10─11  
       ╲ ╱     ╲ ╱
        6       12
         ╲     ╱ ╲ 
```
"""
struct FiniteKagomeXC <: AbstractLattice{2}
    L::Int
    N::Int
    function FiniteKagomeXC(L::Integer, N::Integer = L)
        (L > 0 && N > 0) ||
            throw(ArgumentError("L and N must be strictly positive"))
        mod(L, 3) == 0 || throw(ArgumentError("L must be a multiple of 3"))
        mod(N, L) == 0 || throw(ArgumentError("N must be a multiple of L"))
        return new(L, N)
    end
end

Base.isfinite(::Type{FiniteKagomeXC}) = true
function Base.axes(lattice::FiniteKagomeXC)
    return ((-typemax(Int)):typemax(Int), 1:(2 * (lattice.N ÷ lattice.L)))
end

############################################################################################
# Shared helpers
############################################################################################

# TODO: implement proper bounds checking
function Base.checkbounds(
        ::Type{Bool},
        ::Union{InfiniteKagomeYC, FiniteKagomeYC, InfiniteKagomeXC, FiniteKagomeXC},
        ::Vararg{Int, 2}
    )
    return true
end

# The coordinate system uses a doubled non-orthogonal (oblique) grid in which one step
# equals one kagome bond. Both YC and XC share the same linearization formula.
#
# For a unit cell (k, n) where k ∈ 1:L÷3 indexes around the circumference and n indexes
# along the chain, the sites have doubled coordinates:
#   YC: A at (2k-1, 2n-1), B at (2k, 2n-1), C at (2k-1, 2n)
#   XC: A at (2k-1, 2n-1), C at (2k, 2n-1), B at (2k-1, 2n)
#
# The first coordinate wraps with period 2*(L÷3). The linear index formula is
#   (n - 1)*L + 3*(k - 1) + s,  s ∈ {1, 2, 3}
# where s=1 for site type at (odd,odd), s=2 at (even,odd), s=3 at (odd,even).

const _KagomeLattice = Union{InfiniteKagomeYC, FiniteKagomeYC, InfiniteKagomeXC, FiniteKagomeXC}

function linearize_index(lattice::_KagomeLattice, i::Int, j::Int)
    period = 2 * (lattice.L ÷ 3)
    i_w = mod1(i, period)
    k = (i_w + 1) ÷ 2        # unit cell index around circumference
    n = isodd(j) ? (j + 1) ÷ 2 : j ÷ 2  # unit cell index along chain
    s = if isodd(i_w) && isodd(j)
        1
    elseif isodd(i_w) && iseven(j)
        2
    elseif iseven(i_w) && isodd(j)
        3
    else
        throw(ArgumentError(lazy"invalid site for $lattice: ($i, $j)"))
    end
    return (n - 1) * lattice.L + 3 * (k - 1) + s
end

############################################################################################
# Vertices
############################################################################################

function vertices(lattice::Union{InfiniteKagomeYC, FiniteKagomeYC})
    M = lattice.L ÷ 3
    P = lattice.N ÷ lattice.L
    V = LatticePoint{2, typeof(lattice)}[]
    for n in 1:P, k in 1:M
        push!(V, LatticePoint((2k - 1, 2n - 1), lattice))  # A
        push!(V, LatticePoint((2k, 2n - 1), lattice))  # B
        push!(V, LatticePoint((2k - 1, 2n), lattice))  # C
    end
    return V
end

function vertices(lattice::Union{InfiniteKagomeXC, FiniteKagomeXC})
    M = lattice.L ÷ 3
    P = lattice.N ÷ lattice.L
    V = LatticePoint{2, typeof(lattice)}[]
    for n in 1:P, k in 1:M
        push!(V, LatticePoint((2k - 1, 2n - 1), lattice))  # A
        push!(V, LatticePoint((2k - 1, 2n), lattice))  # B
        push!(V, LatticePoint((2k, 2n - 1), lattice))  # C
    end
    return V
end

############################################################################################
# Nearest neighbours
############################################################################################
# All kagome bonds have unit length in the doubled oblique coordinate system.
# The three bond direction types are:
#   YC: d1=(+1,0) for A–B, d2=(0,+1) for A–C, d3=(−1,+1) for B–C
#   XC: d1=(0,+1) for A–B, d2=(+1,0) for A–C, d3=(+1,−1) for B–C
#
# Each site emits exactly 2 bonds (the remaining 2 are received from neighbours):
#   YC: A emits d1,d2;  B emits d1,d3;  C emits d2,d3
#   XC: A emits d1,d2;  C emits d2,d3;  B emits d1,d3
#
# For finite lattices only C (YC) / B (XC) sites at the last chain unit cell omit
# their chain-crossing bonds.

function nearest_neighbours(lattice::InfiniteKagomeYC)
    V = vertices(lattice)
    neighbours = Pair{eltype(V), eltype(V)}[]
    for v in V
        s = mod1(linearize_index(v), 3)
        if s == 1  # A
            push!(neighbours, v => v + (1, 0))
            push!(neighbours, v => v + (0, 1))
        elseif s == 2  # B
            push!(neighbours, v => v + (1, 0))
            push!(neighbours, v => v + (-1, 1))
        else  # C
            push!(neighbours, v => v + (0, 1))
            push!(neighbours, v => v + (-1, 1))
        end
    end
    return neighbours
end

function nearest_neighbours(lattice::FiniteKagomeYC)
    V = vertices(lattice)
    P = lattice.N ÷ lattice.L
    neighbours = Pair{eltype(V), eltype(V)}[]
    for v in V
        s = mod1(linearize_index(v), 3)
        if s == 1  # A: both bonds stay within the unit cell
            push!(neighbours, v => v + (1, 0))
            push!(neighbours, v => v + (0, 1))
        elseif s == 2  # B: d1 wraps in i, d3 stays within unit cell
            push!(neighbours, v => v + (1, 0))
            push!(neighbours, v => v + (-1, 1))
        else  # C: both bonds cross to the next chain unit cell
            n = v.coordinates[2] ÷ 2  # C has j = 2n (even)
            if n < P
                push!(neighbours, v => v + (0, 1))
                push!(neighbours, v => v + (-1, 1))
            end
        end
    end
    return neighbours
end

function nearest_neighbours(lattice::InfiniteKagomeXC)
    V = vertices(lattice)
    neighbours = Pair{eltype(V), eltype(V)}[]
    for v in V
        s = mod1(linearize_index(v), 3)
        if s == 1  # A
            push!(neighbours, v => v + (1, 0))   # A → C(k,n)
            push!(neighbours, v => v + (0, 1))   # A → B(k,n)
        elseif s == 2  # C: j = 2n-1 (odd)
            push!(neighbours, v => v + (1, 0))   # C → A(k+1,n) [wraps in i]
            push!(neighbours, v => v + (1, -1))   # C → B(k+1,n-1) [wraps in i]
        else  # B: j = 2n (even)
            push!(neighbours, v => v + (0, 1))   # B → A(k,n+1)
            push!(neighbours, v => v + (1, -1))   # B → C(k,n)
        end
    end
    return neighbours
end

function nearest_neighbours(lattice::FiniteKagomeXC)
    V = vertices(lattice)
    P = lattice.N ÷ lattice.L
    neighbours = Pair{eltype(V), eltype(V)}[]
    for v in V
        s = mod1(linearize_index(v), 3)
        j = v.coordinates[2]
        if s == 1  # A: both bonds stay within the unit cell
            push!(neighbours, v => v + (1, 0))   # A → C(k,n)
            push!(neighbours, v => v + (0, 1))   # A → B(k,n)
        elseif s == 2  # C: j = 2n-1 (odd)
            push!(neighbours, v => v + (1, 0))   # C → A(k+1,n) [wraps in i]
            n = (j + 1) ÷ 2
            n > 1 && push!(neighbours, v => v + (1, -1))  # C → B(k+1,n-1) [n≥2]
        else  # B: j = 2n (even)
            n = j ÷ 2
            n < P && push!(neighbours, v => v + (0, 1))  # B → A(k,n+1)
            push!(neighbours, v => v + (1, -1))            # B → C(k,n)
        end
    end
    return neighbours
end

############################################################################################
# Geometry (norm)
############################################################################################
# In the doubled oblique coordinate system, the Cartesian positions are:
#   YC: i-axis along a₁=(1,0), j-axis along a₂=(cos60°, sin60°)
#       → x = i + j/2,  y = j√3/2
#   XC: i-axis along a₂=(cos60°, sin60°), j-axis along a₁=(1,0)
#       → x = j + i/2,  y = i√3/2
# The circumference (i-axis) wraps with period 2*(L÷3); the minimum image is taken.

function LinearAlgebra.norm(p::LatticePoint{2, <:Union{InfiniteKagomeYC, FiniteKagomeYC}})
    i, j = p.coordinates
    period = 2 * (p.lattice.L ÷ 3)
    return minimum(-1:1) do n
        iw = i + n * period
        sqrt((iw + j / 2)^2 + (j * sqrt(3) / 2)^2)
    end
end

function LinearAlgebra.norm(p::LatticePoint{2, <:Union{InfiniteKagomeXC, FiniteKagomeXC}})
    i, j = p.coordinates
    period = 2 * (p.lattice.L ÷ 3)
    return minimum(-1:1) do n
        iw = i + n * period
        sqrt((j + iw / 2)^2 + (iw * sqrt(3) / 2)^2)
    end
end
