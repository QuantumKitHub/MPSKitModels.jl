# function z2_ising_ham(; J=-1, lambda=0.5)
#     sp = Rep[ℤ₂](0 => 1, 1 => 1)

#     sz = TensorMap(zeros, ComplexF64, sp, sp)
#     blocks(sz)[ℤ₂(1)] .= 1
#     blocks(sz)[ℤ₂(0)] .= -1

#     sx = TensorMap(ones, ComplexF64, sp * Rep[ℤ₂](1 => 1), sp)
#     @tensor nn[-1 -2; -3 -4] := sx[-1 1; -3] * conj(sx[-4 1; -2])

#     return MPOHamiltonian(J * nn / 4) + MPOHamiltonian(lambda * sz / 2)
# end

@deprecate z2_ising_ham(; J=-1, lambda=0.5) transverse_field_ising(ComplexF64, ℤ₂; J=-J,
                                                                   hx=lambda)

@deprecate nonsym_ising_ham(; J=-1, spin=1 // 2, lambda=0.5, longit=0.0) transverse_field_ising(;
                                                                                                J=-J,
                                                                                                spin=spin,
                                                                                                hx=lambda,
                                                                                                hz=longit)

# function nonsym_ising_ham(; J=-1, spin=1 // 2, lambda=0.5, longit=0.0)
#     (sx, _, sz) = nonsym_spintensors(spin)
#     i = first(vertices(InfiniteChain(1)))
#     return MPOHamiltonian(LocalOperator(J * sz ⊗ sz, (i, i + 1)) +
#                           LocalOperator(lambda * sx + longit * sz, (i,)))
# end

# function nonsym_xxz_ham(; spin=1, delta=1, zfield=0.0)
#     (sx, sy, sz, _) = nonsym_spintensors(spin)
#     i = first(vertices(InfiniteChain(1)))
#     return MPOHamiltonian(LocalOperator(sx ⊗ sx + sy ⊗ sy + delta * sz ⊗ sz, (i, i + 1)) +
#                           LocalOperator(zfield * sz, (i,)))
# end

@deprecate nonsym_xxz_ham(; spin=1, delta=1, zfield=0.0) xxz(; spin=spin, Δ=delta,
                                                             hz=zfield)

@deprecate su2_xxx_ham(; spin=1 // 2) xxx(ComplexF64, SU₂; spin=spin)

# function su2_xxx_ham(; spin=1 // 2)
#     #only checked for spin = 1 and spin = 2...
#     ph = Rep[SU₂](spin => 1)

#     Sl1 = TensorMap(ones, ComplexF64, ph, Rep[SU₂](1 => 1) * ph) * sqrt(spin^2 + spin)
#     Sr1 = TensorMap(ones, ComplexF64, Rep[SU₂](1 => 1) * ph, ph) * sqrt(spin^2 + spin)

#     @tensor NN[-1 -2; -3 -4] := Sl1[-1; 2 -3] * Sr1[2 -2; -4]

#     return MPOHamiltonian(NN)
# end

@deprecate u1_xxz_ham(; spin=1, delta=1, zfield=0.0) xxz(ComplexF64, U₁; spin=spin, Δ=delta, hz=zfield)

# function u1_xxz_ham(; spin=1, delta=1, zfield=0.0)
#     (sxd, syd, szd, idd) = spinmatrices(spin)
#     @tensor ham[-1 -2; -3 -4] := sxd[-1, -3] * sxd[-2, -4] + syd[-1, -3] * syd[-2, -4] +
#                                  (delta * szd)[-1, -3] * szd[-2, -4] +
#                                  zfield * 0.5 * szd[-1, -3] * idd[-2, -4] +
#                                  zfield * 0.5 * idd[-1, -3] * szd[-2, -4]

#     indu1map = [Irrep[U₁](v) for v in (-spin):1:spin]
#     pspace = U1Space((v => 1 for v in indu1map))

#     symham = TensorMap(zeros, eltype(ham), pspace * pspace, pspace * pspace)

#     for i in 1:size(ham, 1),
#         j in 1:size(ham, 1),
#         k in 1:size(ham, 1),
#         l in 1:size(ham, 1)

#         if ham[i, j, k, l] != 0
#             copy!(symham[(indu1map[i], indu1map[j], indu1map[end - k + 1],
#                           indu1map[end - l + 1])], ham[i:i, j:j, k:k, l:l])
#         end
#     end

#     return MPOHamiltonian(MPSKit.decompose_localmpo(MPSKit.add_util_leg(symham)))
# end