struct EmptyVal
end

Base.:+(::EmptyVal,::EmptyVal) = EmptyVal();
Base.:+(::EmptyVal,v) = v
Base.:+(v,::EmptyVal) = v

Base.:-(::EmptyVal,::EmptyVal) = EmptyVal();
Base.:-(::EmptyVal,v) = -v
Base.:-(v,::EmptyVal) = v


Base.:*(::EmptyVal,::EmptyVal) = EmptyVal()
Base.:*(::EmptyVal,_) = EmptyVal()
Base.:*(_,::EmptyVal) = EmptyVal()
"""
    Implements
    
    H = E0 + ∑ᵢⱼ ∑ₛ K[i,j] c^{s,+}_i c^{s,-}_j + ∑ᵢⱼₖₗ ∑ₛₜ V[i,j,k,l] c^{s,+}_i c^{t,+}_j c^{t,-}_k c^{s,-}_l

    where s and t are spin indices, which can be up or down. The full hamiltonian has U₁ × SU₂ × FermionParity symmetry.

    This should not be regarded as a state of the art qchem-dmrg code! 
        - No attempt was made to incorporate spacegroup symmetries
        - MPSKit does not contain many required algorithms in qchem (orbital ordering/optimization)
"""
quantum_chemistry_hamiltonian(E0,K,V,Elt=ComplexF64) = mapped_quantum_chemistry_hamiltonian(E0,K,V,Elt)[1]

function mapped_quantum_chemistry_hamiltonian(E0,K,V,Elt=ComplexF64)
    basis_size = size(K,1);
    half_basis_size = Int(floor(basis_size/2));

    # the phsyical space
    psp = Vect[(Irrep[U₁]⊠Irrep[SU₂] ⊠ FermionParity)]((0,0,0)=>1, (1,1//2,1)=>1, (2,0,0)=>1);

    ap = TensorMap(ones,Elt,psp*Vect[(Irrep[U₁]⊠Irrep[SU₂] ⊠ FermionParity)]((-1,1//2,1)=>1),psp);
    blocks(ap)[(U₁(0)⊠SU₂(0)⊠FermionParity(0))] .*= -sqrt(2);
    blocks(ap)[(U₁(1)⊠SU₂(1//2)⊠FermionParity(1))]  .*= 1;


    bm = TensorMap(ones,Elt,psp,Vect[(Irrep[U₁]⊠Irrep[SU₂]⊠FermionParity)]((-1,1//2,1)=>1)*psp);
    blocks(bm)[(U₁(0)⊠SU₂(0)⊠FermionParity(0))] .*= sqrt(2);
    blocks(bm)[(U₁(1)⊠SU₂(1//2)⊠FermionParity(1))] .*= -1;

    # this transposition is easier to reason about in a planar way
    am = transpose(ap',(2,1),(3,));
    bp = transpose(bm',(1,),(3,2));
    ap = transpose(ap,(3,1),(2,));
    bm = transpose(bm,(2,),(3,1));
    
    @plansor b_derp[-1 -2;-3] := bp[1;2 -2]*τ[-3 -1;2 1]
    Lmap_ap_to_bp = inv(ap'*ap)*ap'*b_derp;
    @assert norm(ap*Lmap_ap_to_bp-b_derp) < 1e-12;
    @plansor b_derp[-1 -2;-3] := bm[1;2 -2]*τ[-3 -1;2 1]
    Lmap_am_to_bm = inv(am'*am)*am'*b_derp;
    @assert norm(am*Lmap_am_to_bm-b_derp) < 1e-12;

    Rmap_bp_to_ap = transpose(Lmap_am_to_bm',(2,),(1,));
    Rmap_bm_to_am = transpose(Lmap_ap_to_bp',(2,),(1,));
    @plansor a_derp[-1 -2;-3] := bp[1;-1 2]*Rmap_bp_to_ap[1;3]*τ[2 3;-3 -2]
    @assert norm(a_derp-ap) < 1e-12
    @plansor a_derp[-1 -2;-3] := bm[1;-1 2]*Rmap_bm_to_am[1;3]*τ[2 3;-3 -2]
    @assert norm(a_derp-am) < 1e-12

    h_pm = TensorMap(ones,Elt,psp,psp);
    blocks(h_pm)[(U₁(0)⊠SU₂(0)⊠ FermionParity(0))] .=0;
    blocks(h_pm)[(U₁(1)⊠SU₂(1//2)⊠ FermionParity(1))] .=1;
    blocks(h_pm)[(U₁(2)⊠SU₂(0)⊠ FermionParity(0))] .=2;

    @plansor o_derp[-1 -2;-3 -4] := am[-1 1;-3]*ap[1 -2;-4]
    h_pm_derp = transpose(h_pm,(2,1),());
    Lmap_apam_to_pm = inv(o_derp'*o_derp)*o_derp'*h_pm_derp;
    @assert norm(o_derp*Lmap_apam_to_pm-h_pm_derp) <1e-12;

    @plansor o_derp[-1 -2;-3 -4] := bm[-1;-3 1]*bp[-2;1 -4]
    h_pm_derp2 = transpose(h_pm,(),(2,1));
    Rmap_bpbm_to_pm = h_pm_derp2*o_derp'*inv(o_derp*o_derp');
    @assert norm(transpose(h_pm,(),(2,1))-Rmap_bpbm_to_pm*o_derp) < 1e-12
    
    h_ppmm = h_pm*h_pm-h_pm;

    # ----------------------------------------------------------------------
    # Maps something easier to understand to the corresponding virtual index
    # ----------------------------------------------------------------------

    cnt = 1;
    indmap_1L = fill(0,2,basis_size);
    for i in 1:2, j in 1:basis_size
        cnt += 1
        indmap_1L[i,j] = cnt;
    end

    indmap_1R = fill(0,2,basis_size);
    for i in 1:2, j in 1:basis_size
        cnt += 1
        indmap_1R[i,j] = cnt;
    end

    indmap_2L = fill(0,2,half_basis_size,2,half_basis_size);
    for pm1 in 1:2, i in 1:half_basis_size, pm2 in 1:2, j in i:half_basis_size
        i == j && pm1 == 1 && pm2 == 2 && continue 
        cnt += 1
        indmap_2L[pm1,i,pm2,j] = cnt;
    end

    indmap_2R = fill(0,2,basis_size,2,basis_size);
    for pm1 in 1:2, i in half_basis_size+1:basis_size, pm2 in 1:2, j in i:basis_size
        i == j && pm1 == 1 && pm2 == 2 && continue 
        cnt += 1
        indmap_2R[pm1,i,pm2,j] = cnt;
    end

    hamdat = convert(Array{Any,3},fill(EmptyVal(),basis_size,cnt+1,cnt+1));#Array{Any,3}(missing,basis_size,cnt+2,cnt+2);
    hamdat[:,1,1].+= Elt(1);
    hamdat[:,end,end].+= Elt(1);

    # pure onsite interactions:
    for i in 1:basis_size
        # onsite kinetic part
        hamdat[i,1,end] += K[i,i]*add_util_leg(h_pm);

        # onsite electronic part
        hamdat[i,1,end] += V[i,i,i,i]*add_util_leg(h_ppmm);
    end

    # fill indmap_1L and indmap_1R
    ut = Tensor(ones,oneunit(psp));
    @plansor ut_ap[-1 -2;-3 -4] := ut[-1]*ap[-3 -2;-4];
    @plansor ut_am[-1 -2;-3 -4] := ut[-1]*am[-3 -2;-4];
    @plansor bp_ut[-1 -2;-3 -4] := bp[-1;-3 -2]*conj(ut[-4]);
    @plansor bm_ut[-1 -2;-3 -4] := bm[-1;-3 -2]*conj(ut[-4]);
    for i in 1:basis_size
        hamdat[i,1,indmap_1L[1,i]] += ut_ap;
        hamdat[i,1,indmap_1L[2,i]] += ut_am;
        
        hamdat[i,indmap_1R[1,i],end] += bp_ut
        hamdat[i,indmap_1R[2,i],end] += bm_ut
        
        for loc in i+1:basis_size
            hamdat[loc,indmap_1L[1,i],indmap_1L[1,i]] = Elt(1)
            hamdat[loc,indmap_1L[2,i],indmap_1L[2,i]] = Elt(1)
        end

        for loc in 1:i-1
            hamdat[loc,indmap_1R[1,i],indmap_1R[1,i]] = Elt(1)
            hamdat[loc,indmap_1R[2,i],indmap_1R[2,i]] = Elt(1)
        end
    end

    # indmap_2 onsite part
    # we need pp, mm, pm
    pp_f = isometry(fuse(_lastspace(ap)'*_lastspace(ap)'),_lastspace(ap)'*_lastspace(ap)');
    mm_f = isometry(fuse(_lastspace(am)'*_lastspace(am)'),_lastspace(am)'*_lastspace(am)');
    mp_f = isometry(fuse(_lastspace(am)'*_lastspace(ap)'),_lastspace(am)'*_lastspace(ap)');
    pm_f = isometry(fuse(_lastspace(ap)'*_lastspace(am)'),_lastspace(ap)'*_lastspace(am)');


    @plansor ut_apap[-1 -2;-3 -4] := ut[-1]*ap[-3 1;3]*ap[1 -2;4]*conj(pp_f[-4;3 4]);
    @plansor ut_amam[-1 -2;-3 -4] := ut[-1]*am[-3 1;3]*am[1 -2;4]*conj(mm_f[-4;3 4]);
    @plansor ut_amap[-1 -2;-3 -4] := ut[-1]*am[-3 1;3]*ap[1 -2;4]*conj(mp_f[-4;3 4]);
    @plansor bpbp_ut[-1 -2;-3 -4] := mm_f[-1;1 2]*bp[1;-3 3]*bp[2;3 -2]*conj(ut[-4]);
    @plansor bmbm_ut[-1 -2;-3 -4] := pp_f[-1;1 2]*bm[1;-3 3]*bm[2;3 -2]*conj(ut[-4]);
    @plansor bmbp_ut[-1 -2;-3 -4] := pm_f[-1;1 2]*bm[1;-3 3]*bp[2;3 -2]*conj(ut[-4]);
    
    for i in 1:basis_size
        if i<=half_basis_size
            hamdat[i,1,indmap_2L[1,i,1,i]] += ut_apap;
            hamdat[i,1,indmap_2L[2,i,1,i]] += ut_amap;
            hamdat[i,1,indmap_2L[2,i,2,i]] += ut_amam;
            for loc in i+1:basis_size
                hamdat[loc,indmap_2L[1,i,1,i],indmap_2L[1,i,1,i]] = Elt(1);
                hamdat[loc,indmap_2L[2,i,1,i],indmap_2L[2,i,1,i]] = Elt(1);
                hamdat[loc,indmap_2L[2,i,2,i],indmap_2L[2,i,2,i]] = Elt(1);
            end
        else
            hamdat[i,indmap_2R[1,i,1,i],end] += bpbp_ut;
            hamdat[i,indmap_2R[2,i,1,i],end] += bmbp_ut;
            hamdat[i,indmap_2R[2,i,2,i],end] += bmbm_ut;
            for loc in 1:i-1
                hamdat[loc,indmap_2R[1,i,1,i],indmap_2R[1,i,1,i]] = Elt(1);
                hamdat[loc,indmap_2R[2,i,1,i],indmap_2R[2,i,1,i]] = Elt(1);
                hamdat[loc,indmap_2R[2,i,2,i],indmap_2R[2,i,2,i]] = Elt(1);
            end
        end
    end

    # indmap_2 disconnected part
    iso_pp = isomorphism(_lastspace(ap)',_lastspace(ap)');
    iso_mm = isomorphism(_lastspace(am)',_lastspace(am)');

    @plansor p_ap[-1 -2;-3 -4] := iso_pp[-1;1]*τ[1 2;-3 3]*ap[2 -2;4]*conj(pp_f[-4;3 4]);
    @plansor m_ap[-1 -2;-3 -4] := iso_mm[-1;1]*τ[1 2;-3 3]*ap[2 -2;4]*conj(mp_f[-4;3 4]);
    @plansor p_am[-1 -2;-3 -4] := iso_pp[-1;1]*τ[1 2;-3 3]*am[2 -2;4]*conj(pm_f[-4;3 4]);
    @plansor m_am[-1 -2;-3 -4] := iso_mm[-1;1]*τ[1 2;-3 3]*am[2 -2;4]*conj(mm_f[-4;3 4]);

    @plansor bp_p[-1 -2;-3 -4] := bp[2;-3 3]*iso_mm[1;-4]*τ[4 -2;3 1]*mm_f[-1;2 4]
    @plansor bm_p[-1 -2;-3 -4] := bm[2;-3 3]*iso_mm[1;-4]*τ[4 -2;3 1]*pm_f[-1;2 4]
    @plansor bm_m[-1 -2;-3 -4] := bm[2;-3 3]*iso_pp[1;-4]*τ[4 -2;3 1]*pp_f[-1;2 4]
    @plansor bp_m[-1 -2;-3 -4] := bp[2;-3 3]*iso_pp[1;-4]*τ[4 -2;3 1]*mp_f[-1;2 4]

    for i in 1:basis_size, j in i+1:basis_size
        if j<=half_basis_size
            hamdat[j,indmap_1L[1,i],indmap_2L[1,i,1,j]] += p_ap;
            hamdat[j,indmap_1L[1,i],indmap_2L[1,i,2,j]] += p_am;
            hamdat[j,indmap_1L[2,i],indmap_2L[2,i,1,j]] += m_ap;
            hamdat[j,indmap_1L[2,i],indmap_2L[2,i,2,j]] += m_am;
            for k in j+1:basis_size
                hamdat[k,indmap_2L[1,i,1,j],indmap_2L[1,i,1,j]] = Elt(1);
                hamdat[k,indmap_2L[1,i,2,j],indmap_2L[1,i,2,j]] = Elt(1);
                hamdat[k,indmap_2L[2,i,2,j],indmap_2L[2,i,2,j]] = Elt(1);
                hamdat[k,indmap_2L[2,i,1,j],indmap_2L[2,i,1,j]] = Elt(1);
            end
        end

        if i > half_basis_size
            hamdat[i,indmap_2R[1,i,1,j],indmap_1R[1,j]] += bp_p;
            hamdat[i,indmap_2R[1,i,2,j],indmap_1R[2,j]] += bp_m;
            hamdat[i,indmap_2R[2,i,1,j],indmap_1R[1,j]] += bm_p;
            hamdat[i,indmap_2R[2,i,2,j],indmap_1R[2,j]] += bm_m;
            for k in 1:i-1
                hamdat[k,indmap_2R[1,i,1,j],indmap_2R[1,i,1,j]] = Elt(1);
                hamdat[k,indmap_2R[1,i,2,j],indmap_2R[1,i,2,j]] = Elt(1);
                hamdat[k,indmap_2R[2,i,2,j],indmap_2R[2,i,2,j]] = Elt(1);
                hamdat[k,indmap_2R[2,i,1,j],indmap_2R[2,i,1,j]] = Elt(1);
            end
        end

    end


    # fill in all T terms
    # 1 | 1
    for i in 1:basis_size, j in i+1:basis_size
        # m | p .
        hamdat[j,indmap_1L[2,i],end] += K[j,i]*bp_ut;
        
        # p | . m
        hamdat[j,indmap_1L[1,i],end] += K[i,j]*bm_ut;
    end
    
    
    # fill in all V terms

    # 3|1
    @plansor ppRm[-1 -2;-3 -4] := ut[-1]*ap[1 -2;-4]*h_pm[1;-3]
    @plansor Rpmm[-1 -2;-3 -4] := ut[-1]*h_pm[-2;1]*am[-3 1;-4]
    for i in 1:basis_size,j in i+1:basis_size
        # p p . m | m
        # p p m . | m
        hamdat[i,1,indmap_1R[2,j]] += (V[i,i,j,i]+V[i,i,i,j])*ppRm

        # . p m m | p
        # p . m m | p
        hamdat[i,1,indmap_1R[1,j]] += (V[j,i,i,i]+V[i,j,i,i])*Rpmm
    end
    

    # 1|3
    @plansor ppLm[-1 -2;-3 -4] := bp[-1;1 -2]*h_pm[1;-3]*conj(ut[-4])
    @plansor Lpmm[-1 -2;-3 -4] := bm[-1;-3 1]*h_pm[-2;1]*conj(ut[-4])
    for i in 1:basis_size,j in i+1:basis_size
        # m | p p . m
        # m | p p m .
        hamdat[j,indmap_1L[2,i],end] += (V[j,j,i,j]+V[j,j,j,i])*ppLm;

        # p | . p m m
        # p | p . m m
        hamdat[j,indmap_1L[1,i],end] += (V[j,i,j,j]+V[i,j,j,j])*Lpmm;
    end


    # 2|2
    @plansor __mm[-1 -2;-3 -4] := pp_f[-1;1 2]*bm[1;-3 3]*bm[2;3 -2]*conj(ut[-4])
    @plansor __pp[-1 -2;-3 -4] := mm_f[-1;1 2]*bp[1;-3 3]*bp[2;3 -2]*conj(ut[-4])
    @plansor _p_m[-1 -2;-3 -4] := mp_f[-1;1 2] * bp[1;-3 3]*bm[2;3 -2]*conj(ut[-4])
    @plansor _pm_[-1 -2;-3 -4] := mp_f[-1;1 2]*Lmap_apam_to_pm[1 2]*h_pm[-2;-3]*conj(ut[-4])
    p_m_ = _p_m;
    p__m = _pm_;
    
    for i in 1:half_basis_size, j in i+1:basis_size
        # p p | . . m m
        hamdat[j,indmap_2L[1,i,1,i],end] += V[i,i,j,j]*__mm;

        # m m | p p . .
        hamdat[j,indmap_2L[2,i,2,i],end] += V[j,j,i,i]*__pp;

        # p m | . p . m
        hamdat[j,indmap_2L[2,i,1,i],end] += V[i,j,i,j]*_p_m;
        hamdat[i,1,end] -= V[i,j,i,j]*add_util_leg(h_pm);

        # p m | p . m .
        hamdat[j,indmap_2L[2,i,1,i],end] += V[j,i,j,i]*p_m_;
        hamdat[i,1,end] -= V[j,i,j,i]*add_util_leg(h_pm);
        
        # p m | . p m .
        hamdat[j,indmap_2L[2,i,1,i],end] += V[i,j,j,i]*_pm_;

        # p m | p . . m
        hamdat[j,indmap_2L[2,i,1,i],end] += V[j,i,i,j]*_pm_;
    end

    @plansor __mm[-1 -2;-3 -4] := ut[-1]*am[-3 1;2]*am[1 -2;3]*conj(mm_f[-4;2 3])
    @plansor __pp[-1 -2;-3 -4] := ut[-1]*ap[-3 1;2]*ap[1 -2;3]*conj(pp_f[-4;2 3])
    @plansor _p_m[-1 -2;-3 -4] := ut[-1]*ap[-3 1;2]*am[1 -2;3]*conj(pm_f[-4;2 3])
    p_m_ = _p_m;
    @plansor _pm_[-1 -2;-3 -4] := ut[-1]*h_pm[-2;-3]*(transpose(Rmap_bpbm_to_pm*pm_f',(1,)))[-4]
    p__m = _pm_
    for i in half_basis_size+1:basis_size, j in i+1:basis_size
        #  . . m m | p p
        hamdat[i,1,indmap_2R[1,j,1,j]] += V[j,j,i,i]*__mm;

        # p p . . | m m 
        hamdat[i,1,indmap_2R[2,j,2,j]] += V[i,i,j,j]*__pp;

        # . p . m | p m
        hamdat[i,1,indmap_2R[2,j,1,j]] += V[j,i,j,i]*_p_m;
        hamdat[j,1,end] -= V[j,i,j,i]*add_util_leg(h_pm);

        #  p . m . | p m
        hamdat[i,1,indmap_2R[2,j,1,j]] += V[i,j,i,j]*_p_m;
        hamdat[j,1,end] -= V[i,j,i,j]*add_util_leg(h_pm);
        
        #  . p m . | p m
        hamdat[i,1,indmap_2R[2,j,1,j]] += V[i,j,j,i]*_pm_;

        #  p . . m | p m
        hamdat[i,1,indmap_2R[2,j,1,j]] += V[j,i,i,j]*_pm_;
    end

    # 1|2|1
    @plansor LRmm[-1 -2;-3 -4] := am[1 -2;-4]*bm[-1;-3 1]
    @plansor RLmm[-1 -2;-3 -4] := bm[-1;1 -2]*am[-3 1;-4]

    @plansor ppLR[-1 -2;-3 -4] := ap[1 -2;-4]*bp[-1;-3 1]
    @plansor ppRL[-1 -2;-3 -4] := bp[-1;1 -2]*ap[-3 1;-4]

    @plansor LpRm[-1 -2;-3 -4] := ap[1 -2;-4]*bm[-1;-3 1]
    @plansor pLmR[-1 -2;-3 -4] := ap[1 -2;-4]*bm[-1;-3 1]
    
    @plansor RpLm[-1 -2;-3 -4] := bp[-1;1 -2]*am[-3 1;-4]
    @plansor pRmL[-1 -2;-3 -4] := bp[-1;1 -2]*am[-3 1;-4]

    @plansor LpmR[-1 -2;-3 -4] := h_pm[4;-3]*iso_pp[-1;3]*τ[3 -2;4 -4]
    @plansor pLRm[-1 -2;-3 -4] := iso_pp[-1;1]*τ[1 -2;2 -4]*h_pm[2;-3]

    @plansor RpmL[-1 -2;-3 -4] := h_pm[4;-3]*iso_mm[-1;3]*τ[3 -2;4 -4]
    @plansor pRLm[-1 -2;-3 -4] := iso_mm[-1;1]*τ[1 -2;2 -4]*h_pm[2;-3]
    
    for i in 1:basis_size,j in i+1:basis_size,k in j+1:basis_size
        # L R m m
        hamdat[j,indmap_1L[1,i],indmap_1R[1,k]] += V[i,k,j,j]*LRmm

        # R L m m
        hamdat[j,indmap_1L[1,i],indmap_1R[1,k]] += V[k,i,j,j]*RLmm;
        
        # p p L R
        hamdat[j,indmap_1L[2,i],indmap_1R[2,k]] += V[j,j,i,k]*ppLR

        # p p R L
        hamdat[j,indmap_1L[2,i],indmap_1R[2,k]] += V[j,j,k,i]*ppRL

        # L p R m
        hamdat[j,indmap_1L[1,i],indmap_1R[2,k]] += V[i,j,k,j]*LpRm

        # p L m R
        hamdat[j,indmap_1L[1,i],indmap_1R[2,k]] += V[j,i,j,k]*pLmR

        # R p L m
        hamdat[j,indmap_1L[2,i],indmap_1R[1,k]] += V[k,j,i,j]*RpLm

        # p R m L
        hamdat[j,indmap_1L[2,i],indmap_1R[1,k]] += V[j,k,j,i]*pRmL

        # L p m R
        hamdat[j,indmap_1L[1,i],indmap_1R[2,k]] += V[i,j,j,k]*LpmR

        # p L R m
        hamdat[j,indmap_1L[1,i],indmap_1R[2,k]] += V[j,i,k,j]*pLRm

        # R p m L
        hamdat[j,indmap_1L[2,i],indmap_1R[1,k]] += V[k,j,j,i]*RpmL

        # p R L m
        hamdat[j,indmap_1L[2,i],indmap_1R[1,k]] += V[j,k,i,j]*pRLm
    end
    
    # 2|1|1
    @plansor __mm[-1 -2;-3 -4]:= am[-3 1;2]*am[1 -2;3]*conj(mm_f[-4;2 3])*ut[-1]
    @plansor pp__[-1 -2;-3 -4] := ap[-3 1;2]*ap[1 -2;3]*conj(pp_f[-4;2 3])*ut[-1]
    @plansor pjkm[-1 -2;-3 -4] := ut[-1]*h_pm[-2;-3]*Rmap_bp_to_ap[1;2]*conj(mp_f[-4;1 2])
    @plansor jpmk[-1 -2;-3 -4] := ut[-1]*h_pm[-2;-3]*Rmap_bp_to_ap[1;2]*conj(mp_f[-4;1 2])
    @plansor kpmj[-1 -2;-3 -4] := ut[-1]*h_pm[-2;-3]*Rmap_bm_to_am[1;2]*conj(pm_f[-4;1 2])
    @plansor pkjm[-1 -2;-3 -4] := ut[-1]*h_pm[-2;-3]*Rmap_bm_to_am[1;2]*conj(pm_f[-4;1 2])
    @plansor jpkm[-1 -2;-3 -4] := am[-3 1;2]*ap[1 -2;3]*conj(mp_f[-4;2 3])*ut[-1]
    @plansor pjmk[-1 -2;-3 -4] := am[-3 1;2]*ap[1 -2;3]*conj(mp_f[-4;2 3])*ut[-1]
    @plansor kpjm[-1 -2;-3 -4] := ap[-3 1;2]*am[1 -2;3]*conj(pm_f[-4;2 3])*ut[-1]
    @plansor pkmj[-1 -2;-3 -4] := ap[-3 1;2]*am[1 -2;3]*conj(pm_f[-4;2 3])*ut[-1]
    @plansor kpjm[-1 -2;-3 -4] := am[-3 1;4]*ap[1 -2;5]*τ[4 5;2 3]*conj(pm_f[-4;2 3])*ut[-1]
    pkmj = kpjm

    @plansor LLmR[-1 -2;-3 -4] := pp_f[-1;1 2]*bm[1;-3 3]*τ[3 2;-4 -2]
    @plansor LLRm[-1 -2;-3 -4] := pp_f[-1;1 2]*bm[2;3 -2]*τ[1 3;-3 -4]
    @plansor LpLR[-1 -2;-3 -4] := mp_f[-1;1 2]*bp[1;-3 3]*τ[3 2;-4 -2]
    @plansor LRLm[-1 -2;-3 -4] := mp_f[-1;1 2]*bm[2;3 -2]*τ[1 3;-3 -4]
    @plansor pLRL[-1 -2;-3 -4] := mp_f[-1;1 2]*bp[1;-3 3]*τ[3 2;-4 -2]
    @plansor RLmL[-1 -2;-3 -4] := mp_f[-1;1 2]*bm[2;3 -2]*τ[1 3;-3 -4]
    @plansor RpLL[-1 -2;-3 -4] := mm_f[-1;1 2]*bp[2;3 -2]*τ[1 3;-3 -4]
    @plansor pRLL[-1 -2;-3 -4] := mm_f[-1;1 2]*bp[1;-3 3]*τ[3 2;-4 -2]
    @plansor LpRL[-1 -2;-3 -4] := mp_f[-1;1 2]*Lmap_apam_to_pm[1 2]*ap[-3 -2;-4]
    @plansor pLLR[-1 -2;-3 -4] := mp_f[-1;1 2]*Lmap_apam_to_pm[1 2]*ap[-3 -2;-4]
    @plansor LRmL[-1 -2;-3 -4] := mp_f[-1;1 2]*Lmap_apam_to_pm[1 2]*am[-3 -2;-4]
    @plansor RLLm[-1 -2;-3 -4] := mp_f[-1;1 2]*Lmap_apam_to_pm[1 2]*am[-3 -2;-4]

    for i in half_basis_size+1:basis_size,j in i+1:basis_size,k in j+1:basis_size
        # j k m m
        # k j m m
        hamdat[i,1,indmap_2R[1,j,1,k]] += (V[j,k,i,i]+V[k,j,i,i])*__mm;
        
        # p p j k
        # p p k j
        hamdat[i,1,indmap_2R[2,j,2,k]] += (V[i,i,j,k]+V[i,i,k,j])*pp__;

        # p j k m
        hamdat[i,1,indmap_2R[1,j,2,k]] += V[i,j,k,i]*pjkm
        # j p m k
        hamdat[i,1,indmap_2R[1,j,2,k]] += V[j,i,i,k]*jpmk;

        # p k j m
        hamdat[i,1,indmap_2R[2,j,1,k]] += V[i,k,j,i]*pkjm
        # k p m j
        hamdat[i,1,indmap_2R[2,j,1,k]] += V[k,i,i,j]*kpmj;

        # j p k m
        hamdat[i,1,indmap_2R[1,j,2,k]] += V[j,i,k,i]*jpkm;
        # p j m k
        hamdat[i,1,indmap_2R[1,j,2,k]] += V[i,j,i,k]*pjmk;

        # k p j m
        hamdat[i,1,indmap_2R[2,j,1,k]] += V[k,i,j,i]*kpjm

        # p k m j
        hamdat[i,1,indmap_2R[2,j,1,k]] += V[i,k,i,j]*pkmj
    end
    for i in 1:half_basis_size,j in i+1:basis_size,k in j+1:basis_size
        # L L m R    
        hamdat[j,indmap_2L[1,i,1,i],indmap_1R[2,k]] += V[i,i,j,k]*LLmR

        # L L R m
        hamdat[j,indmap_2L[1,i,1,i],indmap_1R[2,k]] += V[i,i,k,j]*LLRm

        # L p L R
        hamdat[j,indmap_2L[2,i,1,i],indmap_1R[2,k]] += V[i,j,i,k]*LpLR

        # L R L m
        hamdat[j,indmap_2L[2,i,1,i],indmap_1R[1,k]] += V[i,k,i,j]*LRLm

        # p L R L
        hamdat[j,indmap_2L[2,i,1,i],indmap_1R[2,k]] += V[j,i,k,i]*pLRL

        # R L m L
        hamdat[j,indmap_2L[2,i,1,i],indmap_1R[1,k]] += V[k,i,j,i]*RLmL

        # R p L L
        hamdat[j,indmap_2L[2,i,2,i],indmap_1R[1,k]] += V[k,j,i,i]*RpLL

        # p R L L
        hamdat[j,indmap_2L[2,i,2,i],indmap_1R[1,k]] += V[j,k,i,i]*pRLL

        # L p R L
        hamdat[j,indmap_2L[2,i,1,i],indmap_1R[2,k]] += V[i,j,k,i]*LpRL

        # L R m L
        hamdat[j,indmap_2L[2,i,1,i],indmap_1R[1,k]] += V[i,k,j,i]*LRmL
        
        # p L L R
        hamdat[j,indmap_2L[2,i,1,i],indmap_1R[2,k]] += V[j,i,i,k]*pLLR

        # R L L m
        hamdat[j,indmap_2L[2,i,1,i],indmap_1R[1,k]] += V[k,i,i,j]*RLLm
    end


    # 1|1|2
    # (i,j) in indmap_2, 2 onsite
    @plansor jimm[-1 -2;-3 -4] := bm[1;-3 2]*bm[3;2 -2]*τ[4 5;1 3]*pp_f[-1;4 5]*conj(ut[-4])
    @plansor ijmm[-1 -2;-3 -4] := bm[1;-3 2]*bm[3;2 -2]*τ[4 5;1 3]*permute(pp_f,(1,),(3,2))[-1;4 5]*conj(ut[-4])

    @plansor jpim[-1 -2;-3 -4] := bm[1;-3 2]*bp[3;2 -2]*τ[4 5;1 3]*mp_f[-1;4 5]*conj(ut[-4])
    
    @plansor ipjm[-1 -2;-3 -4] := bm[1;-3 2]*bp[3;2 -2]*τ[4 5;1 3]*permute(pm_f,(1,),(3,2))[-1;4 5]*conj(ut[-4])
    
    @plansor ppji[-1 -2;-3 -4] := mm_f[-1;1 2]*bp[1;-3 3]*bp[2;3 -2]*conj(ut[-4])
    @plansor ppij[-1 -2;-3 -4] := permute(mm_f,(1,),(3,2))[-1;1 2]*bp[1;-3 3]*bp[2;3 -2]*conj(ut[-4])

    @plansor jpmi[-1 -2;-3 -4] := ut[-1]*h_pm[-2;-3]*conj(ut[-4])
    @plansor ipmj[-1 -2;-3 -4] := ut[-1]*h_pm[-2;-3]*conj(ut[-4])

    pjmi = jpim;
    pimj = ipjm;
    
    @plansor connect_am_ap[-1 -2;-3 -4] := mp_f[-1;1 2]*Lmap_ap_to_bp[2;1]*h_pm[-2;-3]*conj(ut[-4])
    @plansor connect_ap_am[-1 -2;-3 -4] := pm_f[-1;1 2]*Lmap_am_to_bm[2;1]*h_pm[-2;-3]*conj(ut[-4])

    for i in 1:basis_size,j in i+1:basis_size,k in j+1:basis_size
        k <=half_basis_size || continue

        # j i m m
        hamdat[k,indmap_2L[1,i,1,j],end] += V[j,i,k,k]*jimm
        # i j m m
        hamdat[k,indmap_2L[1,i,1,j],end] += V[i,j,k,k]*ijmm;

        # j p i m
        hamdat[k,indmap_2L[2,i,1,j],end] += V[j,k,i,k]*jpim
        hamdat[k,indmap_2L[2,i,1,j],end] += V[k,j,k,i]*pjmi

        # i p j m
        hamdat[k,indmap_2L[1,i,2,j],end] += V[i,k,j,k]*ipjm
        hamdat[k,indmap_2L[1,i,2,j],end] += V[k,i,k,j]*pimj

        # j p m i
        hamdat[k,indmap_2L[2,i,1,j],end] += V[j,k,k,i]*connect_am_ap
        hamdat[k,indmap_2L[2,i,1,j],end] += V[k,j,i,k]*connect_am_ap
        # i p m j
        hamdat[k,indmap_2L[1,i,2,j],end] += V[i,k,k,j]*connect_ap_am
        hamdat[k,indmap_2L[1,i,2,j],end] += V[k,i,j,k]*connect_ap_am

        # p p j i
        hamdat[k,indmap_2L[2,i,2,j],end] += V[k,k,j,i]*ppji
        # p p i j
        hamdat[k,indmap_2L[2,i,2,j],end] += V[k,k,i,j]*ppij
    end

    @plansor jimm[-1 -2;-3 -4] := iso_pp[-1;1]*ap[-3 2;3]*τ[2 1;4 -2]*conj(pp_f[-4;3 4])
    @plansor ppji[-1 -2;-3 -4] := iso_mm[-1;1]*am[-3 2;3]*τ[2 1;4 -2]*conj(mm_f[-4;3 4])
    @plansor jpim[-1 -2;-3 -4] := iso_mm[-1;1]*ap[-3 2;3]*τ[2 1;4 -2]*conj(pm_f[-4;3 4])
    @plansor ipjm[-1 -2;-3 -4] := iso_pp[-1;1]*τ[-3 1;2 3]*am[3 -2;4]*conj(pm_f[-4;2 4])
    
    @plansor jpmi[-1 -2;-3 -4] := bp[-1;-3 -2]*(transpose(Rmap_bpbm_to_pm*pm_f',(1,)))[-4]
    @plansor ipmj[-1 -2;-3 -4] := bm[-1;-3 -2]*(transpose(Rmap_bpbm_to_pm*pm_f',(1,)))[-4]
    for i in 1:basis_size,j in i+1:basis_size,k in j+1:basis_size
        k > half_basis_size || continue

        # j i m m
        # i j m m
        hamdat[j,indmap_1L[1,i],indmap_2R[2,k,2,k]] += (V[j,i,k,k]+V[i,j,k,k])*jimm

        # p p j i
        # p p i j
        hamdat[j,indmap_1L[2,i],indmap_2R[1,k,1,k]] += (V[k,k,j,i]+V[k,k,i,j])*ppji
        
        # j p i m
        # p j m i
        hamdat[j,indmap_1L[2,i],indmap_2R[2,k,1,k]] += (V[j,k,i,k]+V[k,j,k,i])*jpim

        # i p j m
        # p i m j
        hamdat[j,indmap_1L[1,i],indmap_2R[2,k,1,k]] += (V[i,k,j,k]+V[k,i,k,j])*ipjm

        # j p m i
        # p j i m
        hamdat[j,indmap_1L[2,i],indmap_2R[2,k,1,k]] += (V[k,j,i,k]+V[j,k,k,i])*jpmi

        # i p m j
        # p i j m
        hamdat[j,indmap_1L[1,i],indmap_2R[2,k,1,k]] += (V[i,k,k,j]+V[k,i,j,k])*ipmj
    end

    # 1|1|1|1
    # (i,j) in indmap_2, 1 in indmap_4, 1 onsite
    @plansor pjiR[-1 -2;-3 -4] := mp_f[-1;1 2]*Lmap_ap_to_bp[2;1]*ap[-3 -2;-4]
    @plansor jpRi[-1 -2;-3 -4] := mp_f[-1;1 2]*Lmap_ap_to_bp[2;1]*ap[-3 -2;-4]
    @plansor pijR[-1 -2;-3 -4] := pm_f[-1;1 2]*Lmap_am_to_bm[2;1]*ap[-3 -2;-4]
    @plansor ipRj[-1 -2;-3 -4] := pm_f[-1;1 2]*Lmap_am_to_bm[2;1]*ap[-3 -2;-4]
    @plansor Rjim[-1 -2;-3 -4] := mp_f[-1;1 2]*Lmap_ap_to_bp[2;1]*am[-3 -2;-4]
    @plansor jRmi[-1 -2;-3 -4] := mp_f[-1;1 2]*Lmap_ap_to_bp[2;1]*am[-3 -2;-4]
    @plansor Rijm[-1 -2;-3 -4] := pm_f[-1;1 2]*Lmap_am_to_bm[2;1]*am[-3 -2;-4]
    @plansor iRmj[-1 -2;-3 -4] := pm_f[-1;1 2]*Lmap_am_to_bm[2;1]*am[-3 -2;-4]
    @plansor jimR[-1 -2;-3 -4] := pp_f[-1;1 2]*τ[3 2;-4 -2]*bm[1;-3 3]
    @plansor ijRm[-1 -2;-3 -4] := permute(pp_f,(1,),(3,2))[-1;1 2]*τ[1 3;-3 -4]*bm[2;3 -2]
    @plansor ijmR[-1 -2;-3 -4] := permute(pp_f,(1,),(3,2))[-1;1 2]*τ[3 2;-4 -2]*bm[1;-3 3]
    @plansor jiRm[-1 -2;-3 -4] := pp_f[-1;1 2]*τ[1 3;-3 -4]*bm[2;3 -2]
    @plansor jpiR[-1 -2;-3 -4] := mp_f[-1;1 2]*τ[3 2;-4 -2]*bp[1;-3 3]
    @plansor pjRi[-1 -2;-3 -4] := mp_f[-1;1 2]*τ[3 2;-4 -2]*bp[1;-3 3];
    @plansor ipjR[-1 -2;-3 -4] := permute(pm_f,(1,),(3,2))[-1;1 2]*τ[3 2;-4 -2]*bp[1;-3 3]
    @plansor piRj[-1 -2;-3 -4] := permute(pm_f,(1,),(3,2))[-1;1 2]*τ[3 2;-4 -2]*bp[1;-3 3];
    @plansor jRim[-1 -2;-3 -4] := mp_f[-1;1 2]*τ[1 3;-3 -4]*bm[2;3 -2]
    @plansor Rjmi[-1 -2;-3 -4] := mp_f[-1;1 2]*τ[1 3;-3 -4]*bm[2;3 -2]
    @plansor iRjm[-1 -2;-3 -4] := permute(pm_f,(1,),(3,2))[-1;1 2]*τ[1 3;-3 -4]*bm[2;3 -2]
    @plansor Rimj[-1 -2;-3 -4] := permute(pm_f,(1,),(3,2))[-1;1 2]*τ[1 3;-3 -4]*bm[2;3 -2]
    @plansor Rpji[-1 -2;-3 -4] := mm_f[-1;1 2]*τ[1 3;-3 -4]*bp[2;3 -2]
    @plansor pRij[-1 -2;-3 -4] := permute(mm_f,(1,),(3,2))[-1;1 2]*τ[3 2;-4 -2]*bp[1;-3 3]
    @plansor Rpij[-1 -2;-3 -4] := permute(mm_f,(1,),(3,2))[-1;1 2]*τ[1 3;-3 -4]*bp[2;3 -2]
    @plansor pRji[-1 -2;-3 -4] := mm_f[-1;1 2]*τ[3 2;-4 -2]*bp[1;-3 3]
    @plansor kjil[-1 -2;-3 -4] := bp[-1;-3 -2]*Rmap_bp_to_ap[1;2]*conj(mp_f[-4;1 2])
    @plansor jkli[-1 -2;-3 -4] := bp[-1;-3 -2]*Rmap_bp_to_ap[1;2]*conj(mp_f[-4;1 2])
    @plansor ljik[-1 -2;-3 -4] := bp[-1;-3 -2]*Rmap_bm_to_am[1;2]*conj(pm_f[-4;1 2])
    @plansor jlki[-1 -2;-3 -4] := bp[-1;-3 -2]*Rmap_bm_to_am[1;2]*conj(pm_f[-4;1 2])
    @plansor kijl[-1 -2;-3 -4] := bm[-1;-3 -2]*Rmap_bp_to_ap[1;2]*conj(mp_f[-4;1 2])
    @plansor iklj[-1 -2;-3 -4] := bm[-1;-3 -2]*Rmap_bp_to_ap[1;2]*conj(mp_f[-4;1 2])
    @plansor lijk[-1 -2;-3 -4] := bm[-1;-3 -2]*Rmap_bm_to_am[1;2]*conj(pm_f[-4;1 2])
    @plansor ilkj[-1 -2;-3 -4] := bm[-1;-3 -2]*Rmap_bm_to_am[1;2]*conj(pm_f[-4;1 2])
    @plansor jikl[-1 -2;-3 -4] := iso_pp[-1;1]*ap[2 -2;3]*τ[1 2;-3 4]*conj(pp_f[-4;4 3])
    ijlk = jikl
    @plansor likj[-1 -2;-3 -4] := iso_pp[-1;1]*am[2 -2;3]*τ[1 2;-3 4]*conj(pm_f[-4;4 3])
    iljk = likj
    @plansor jkil[-1 -2;-3 -4] := iso_mm[-1;1]*ap[2 -2;3]*τ[1 2;-3 4]*conj(mp_f[-4;4 3])
    kjli = jkil
    @plansor lkij[-1 -2;-3 -4] := iso_mm[-1;1]*am[2 -2;3]*τ[1 2;-3 4]*conj(mm_f[-4;4 3])
    klji = lkij
    @plansor ijkl[-1 -2;-3 -4] := iso_pp[-1;1]*ap[-3 2;3]*τ[2 1;4 -2]*conj(pp_f[-4;3 4])
    jilk = ijkl
    @plansor ljki[-1 -2;-3 -4] := iso_mm[-1;1]*ap[-3 2;3]*τ[2 1;4 -2]*conj(pm_f[-4;3 4])
    jlik = ljki
    @plansor ikjl[-1 -2;-3 -4] := iso_pp[-1;1]*am[-3 2;3]*τ[2 1;4 -2]*conj(mp_f[-4;3 4])
    kilj = ikjl;
    @plansor lkji[-1 -2;-3 -4] := iso_mm[-1;1]*am[-3 2;3]*τ[2 1;4 -2]*conj(mm_f[-4;3 4])
    klij = lkji

    for i in 1:basis_size,j in i+1:basis_size,k in j+1:half_basis_size,l in k+1:basis_size
        # p j i R
        hamdat[k,indmap_2L[2,i,1,j],indmap_1R[2,l]] += V[k,j,i,l]*pjiR
        # j p R i
        hamdat[k,indmap_2L[2,i,1,j],indmap_1R[2,l]] += V[j,k,l,i]*jpRi
        # p i j R
        hamdat[k,indmap_2L[1,i,2,j],indmap_1R[2,l]] += V[k,i,j,l]*pijR
        # i p R j
        hamdat[k,indmap_2L[1,i,2,j],indmap_1R[2,l]] += V[i,k,l,j]*ipRj
        
        # R j i m
        hamdat[k,indmap_2L[2,i,1,j],indmap_1R[1,l]] += V[l,j,i,k]*Rjim        
        # j R m i
        hamdat[k,indmap_2L[2,i,1,j],indmap_1R[1,l]] += V[j,l,k,i]*jRmi
        # R i j m
        hamdat[k,indmap_2L[1,i,2,j],indmap_1R[1,l]] += V[l,i,j,k]*Rijm        
        # i R m j
        hamdat[k,indmap_2L[1,i,2,j],indmap_1R[1,l]] += V[i,l,k,j]*iRmj
    
        # j i m R
        hamdat[k,indmap_2L[1,i,1,j],indmap_1R[2,l]] += V[j,i,k,l]*jimR
        # i j R m
        hamdat[k,indmap_2L[1,i,1,j],indmap_1R[2,l]] += V[i,j,l,k]*ijRm

        # i j m R
        hamdat[k,indmap_2L[1,i,1,j],indmap_1R[2,l]] += V[i,j,k,l]*ijmR
        # j i R m
        hamdat[k,indmap_2L[1,i,1,j],indmap_1R[2,l]] += V[j,i,l,k]*jiRm

        # j p i R
        hamdat[k,indmap_2L[2,i,1,j],indmap_1R[2,l]] += V[j,k,i,l]*jpiR
        # p j R i
        hamdat[k,indmap_2L[2,i,1,j],indmap_1R[2,l]] += V[k,j,l,i] * pjRi

        # i p j R
        hamdat[k,indmap_2L[1,i,2,j],indmap_1R[2,l]] += V[i,k,j,l]*ipjR
        # p i R j
        hamdat[k,indmap_2L[1,i,2,j],indmap_1R[2,l]] += V[k,i,l,j] * piRj

        # j R i m
        hamdat[k,indmap_2L[2,i,1,j],indmap_1R[1,l]] += V[j,l,i,k] * jRim
        # R j m i
        hamdat[k,indmap_2L[2,i,1,j],indmap_1R[1,l]] += V[l,j,k,i]*Rjmi

        # i R j m
        hamdat[k,indmap_2L[1,i,2,j],indmap_1R[1,l]] += V[i,l,j,k] * iRjm
        # R i m j
        hamdat[k,indmap_2L[1,i,2,j],indmap_1R[1,l]] += V[l,i,k,j]*Rimj

        # R p j i
        hamdat[k,indmap_2L[2,i,2,j],indmap_1R[1,l]] += V[l,k,j,i]*Rpji
        # p R i j
        hamdat[k,indmap_2L[2,i,2,j],indmap_1R[1,l]] += V[k,l,i,j]*pRij

        # R p i j
        hamdat[k,indmap_2L[2,i,2,j],indmap_1R[1,l]] += V[l,k,i,j]*Rpij
        # p R j i
        hamdat[k,indmap_2L[2,i,2,j],indmap_1R[1,l]] += V[k,l,j,i]*pRji
    end
    
    for i in 1:basis_size,j in i+1:basis_size,k in j+1:basis_size,l in k+1:basis_size
        k > half_basis_size || continue;

        hamdat[j,indmap_1L[2,i],indmap_2R[1,k,2,l]] += (V[k,j,i,l]+V[j,k,l,i])*kjil
        hamdat[j,indmap_1L[2,i],indmap_2R[2,k,1,l]] += (V[l,j,i,k]+V[j,l,k,i])*ljik
        hamdat[j,indmap_1L[1,i],indmap_2R[1,k,2,l]] += (V[k,i,j,l]+V[i,k,l,j])*kijl
        hamdat[j,indmap_1L[1,i],indmap_2R[2,k,1,l]] += (V[l,i,j,k]+V[i,l,k,j])*lijk
        hamdat[j,indmap_1L[1,i],indmap_2R[2,k,2,l]] += (V[j,i,k,l]+V[i,j,l,k])*jikl
        hamdat[j,indmap_1L[1,i],indmap_2R[2,k,1,l]] += (V[l,i,k,j]+V[i,l,j,k])*likj
        hamdat[j,indmap_1L[2,i],indmap_2R[1,k,2,l]] += (V[j,k,i,l]+V[k,j,l,i])*jkil
        hamdat[j,indmap_1L[2,i],indmap_2R[1,k,1,l]] += (V[l,k,i,j]+V[k,l,j,i])*lkij;
        hamdat[j,indmap_1L[1,i],indmap_2R[2,k,2,l]] += (V[i,j,k,l]+V[j,i,l,k])*ijkl;
        hamdat[j,indmap_1L[2,i],indmap_2R[2,k,1,l]] += (V[l,j,k,i]+V[j,l,i,k])*ljki
        hamdat[j,indmap_1L[1,i],indmap_2R[1,k,2,l]] += (V[i,k,j,l]+V[k,i,l,j])*ikjl
        hamdat[j,indmap_1L[2,i],indmap_2R[1,k,1,l]] += (V[l,k,j,i]+V[k,l,i,j])*lkji
    end


    map!(x->x == EmptyVal() ? Elt(0) : x,hamdat,hamdat);
    th = MPOHamiltonian(convert(Array{Union{Elt,typeof(ut_ap)},3},hamdat));


    th+fill(Elt(E0)/basis_size,basis_size),
        (indmap_1L, indmap_1R, indmap_2L, indmap_2R)
end