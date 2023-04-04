function nonsym_xxz_ham(; spin=1, delta=1, zfield=0.0)
    (sx, sy, sz, _) = nonsym_spintensors(spin)
    return MPOHamiltonian(LocalOperator(sx ⊗ sx + sy ⊗ sy + delta * sz ⊗ sz, (1, 2)) +
                          LocalOperator(zfield * sz, (1,)))
end

function su2_xxx_ham(; spin=1 // 2)
    #only checked for spin = 1 and spin = 2...
    ph = Rep[SU₂](spin => 1)

    Sl1 = TensorMap(ones, ComplexF64, ph, Rep[SU₂](1 => 1) * ph) * sqrt(spin^2 + spin)
    Sr1 = TensorMap(ones, ComplexF64, Rep[SU₂](1 => 1) * ph, ph) * sqrt(spin^2 + spin)

    @tensor NN[-1 -2; -3 -4] := Sl1[-1; 2 -3] * Sr1[2 -2; -4]

    return MPOHamiltonian(NN)
end

function u1_xxz_ham(; spin=1, delta=1, zfield=0.0)
    (sxd, syd, szd, idd) = spinmatrices(spin)
    @tensor ham[-1 -2; -3 -4] := sxd[-1, -3] * sxd[-2, -4] + syd[-1, -3] * syd[-2, -4] +
                                 (delta * szd)[-1, -3] * szd[-2, -4] +
                                 zfield * 0.5 * szd[-1, -3] * idd[-2, -4] +
                                 zfield * 0.5 * idd[-1, -3] * szd[-2, -4]

    indu1map = [Irrep[U₁](v) for v in (-spin):1:spin]
    pspace = U1Space((v => 1 for v in indu1map))

    symham = TensorMap(zeros, eltype(ham), pspace * pspace, pspace * pspace)

    for i in 1:size(ham, 1),
        j in 1:size(ham, 1),
        k in 1:size(ham, 1),
        l in 1:size(ham, 1)

        if ham[i, j, k, l] != 0
            copy!(symham[(indu1map[i], indu1map[j], indu1map[end - k + 1],
                          indu1map[end - l + 1])], ham[i:i, j:j, k:k, l:l])
        end
    end

    return MPOHamiltonian(MPSKit.decompose_localmpo(MPSKit.add_util_leg(symham)))
end

"
A finite ladder: obc in the x direction and pbc in the y direction

Step 1: constructing a vector containing all bonds of the lattice: bonds

Step 2: summing up all two-site operators (opp):
        H = H + LocalOperator(opp, (bond.first,bond.second)) 
"
function nonsym_xxz_ladder_finite(; Nx::Int=1, Ny::Int=4, spin=1 // 2, delta=1)
    #-------------Lattice info-------------------------
    numbonds = Ny <= 2 ? Nx * (Ny - 1) + (Nx - 1) * Ny : Nx * Ny + (Nx - 1) * Ny
    bonds = Vector{Pair{Int,Int}}(undef, 0)
    for x in 1:Nx, y in 1:Ny
        i = (x - 1) * Ny + y
        if !(Ny <= 2 && y == Ny)
            iy = (x - 1) * Ny + mod1(y + 1, Ny)
            a, b = sort([i, iy]; rev=false)
            push!(bonds, Pair(a, b))
        end

        if x != Nx
            ix = x * Ny + y
            a, b = i, ix
            push!(bonds, Pair(a, b))
        end
    end
    sort!(bonds)
    @assert length(bonds) == numbonds "lattice construction error!"

    #--------------sum of all local opps----------------
    (sx, sy, sz, _) = nonsym_spintensors(spin)
    ham_bond = sx ⊗ sx + sy ⊗ sy + delta * sz ⊗ sz
    all_opp = SumOfLocalOperators()
    for bond in bonds
        all_opp = all_opp + LocalOperator(ham_bond, (bond.first, bond.second))
    end

    return MPOHamiltonian(all_opp, Nx * Ny)
end

"
An infinite ladder: infinite in the x direction and pbc in the y direction

Step 1: constructing a vector containing all bonds of the lattice: bonds

Step 2: summing up all two-site operators (opp):
        H = H + LocalOperator(opp, (bond.first,bond.second)) 
"
function nonsym_xxz_ladder_infinite(; Ny::Int=4, spin=1 // 2, delta=1)
    #-------------Lattice info-------------------------
    bonds = Vector{Pair{Int,Int}}(undef, 0)
    for y in 1:Ny
        if !(Ny <= 2 && y == Ny)
            iy = mod1(y + 1, Ny)
            a, b = sort([y, iy]; rev=false)
            push!(bonds, Pair(a, b))
        end

        push!(bonds, Pair(y, Ny + y))
    end
    sort!(bonds)

    #--------------sum of all local opps----------------
    (sx, sy, sz, _) = nonsym_spintensors(spin)
    ham_bond = sx ⊗ sx + sy ⊗ sy + delta * sz ⊗ sz
    all_opp = SumOfLocalOperators()
    for bond in bonds
        all_opp = all_opp + LocalOperator(ham_bond, (bond.first, bond.second))
    end

    return MPOHamiltonian(all_opp, Ny)
end

"""
γ is the interchain coupling strength
"""
function su2_xxx_ladder(; Ny=4, spin=1 // 2, γ=1)
    ph = Rep[SU₂](spin => 1)
    Sl1 = TensorMap(ones, ComplexF64, ph, Rep[SU₂](1 => 1) * ph) * sqrt(spin^2 + spin)
    Sr1 = TensorMap(ones, ComplexF64, Rep[SU₂](1 => 1) * ph, ph) * sqrt(spin^2 + spin)
    @tensor NN[-1 -2; -3 -4] := Sl1[-1; 2 -3] * Sr1[2 -2; -4]

    all_opp = SumOfLocalOperators()

    for y in 1:Ny
        if !(Ny <= 2 && y == Ny)
            iy = mod1(y + 1, Ny)
            a, b = sort([y, iy]; rev=false)
            all_opp += LocalOperator(NN, (a, b))
        end

        all_opp += LocalOperator(NN * γ, (y, Ny + y))
    end

    return MPOHamiltonian(all_opp, Ny)
end
