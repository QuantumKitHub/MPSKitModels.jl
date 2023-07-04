using PrecompileTools

@compile_workload begin
    # ========== Hamiltonians ==========
    for symmetry in (Trivial, Z2Irrep, fℤ₂)
        H = transverse_field_ising(symmetry)
    end
    
    H = kitaev_model()
    
    for symmetry in (Trivial, U1Irrep, SU2Irrep)
        H = heisenberg_XXX(symmetry)
        H = bilinear_biquadratic_model(symmetry)
    end
    for symmetry in (Trivial, U1Irrep)
        H = heisenberg_XXZ(symmetry)
    end
    H = heisenberg_XYZ()
    
    H = hubbard_model(Trivial, Trivial)
    H = hubbard_model(U1Irrep, SU2Irrep)
    
    H = bose_hubbard_model()
    H = bose_hubbard_model(U1Irrep)
    
    # ========== Transfer matrices ==========
    T = classical_ising()
    T = classical_ising(Z2Irrep)
    
    for symmetry in (Trivial, U1Irrep, CU1Irrep)
        T = sixvertex(symmetry)
    end
    
    T = hard_hexagon()
    
    T = qstate_clock()
end