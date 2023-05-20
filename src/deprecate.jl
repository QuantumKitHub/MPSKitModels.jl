@deprecate z2_ising_ham(; J=-1, lambda=0.5) transverse_field_ising(ComplexF64, ℤ₂; J=-J,
                                                                   hx=lambda)
@deprecate(nonsym_ising_ham(; J=-1, spin=1 // 2, lambda=0.5, longit=0.0),
           transverse_field_ising(; J=-J, spin=spin, hx=lambda, hz=longit))
           
@deprecate(nonsym_xxz_ham(; spin=1, delta=1, zfield=0.0),
           xxz(; spin=spin, Δ=delta, hz=zfield))
@deprecate su2_xxx_ham(; spin=1 // 2) xxx(ComplexF64, SU₂; spin=spin)
@deprecate(u1_xxz_ham(; spin=1, delta=1, zfield=0.0),
           xxz(ComplexF64, U₁; spin=spin, Δ=delta, hz=zfield))
@deprecate(nonsym_xxz_ladder_infinite(; Ny::Int=4, spin=1//2, delta=1),
          xxz(ComplexF64, ℤ₁, InfiniteCylinder(Ny), Δ=delta, spin=spin))

@deprecate nonsym_sixvertex_mpo(a=1.0, b=1.0, c=1.0) sixvertex(; a=a, b=b, c=c)
@deprecate u1_sixvertex_mpo(a=1.0, b=1.0, c=1.0) sixvertex(ComplexF64, U₁; a=a, b=b, c=c)
@deprecate cu1_sixvertex_mpo(a=1.0, b=1.0, c=1.0) sixvertex(ComplexF64, CU₁; a=a, b=b, c=c)

@deprecate nonsym_qstateclock_mpo(beta::Float64, q::Int) qstate_clock(; beta=beta, q=q)