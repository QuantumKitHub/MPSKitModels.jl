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
        - MPOHamiltonian is not well suited for quantum chemistry
"""
quantum_chemistry_hamiltonian(E0,K,V,Elt=ComplexF64) = mapped_quantum_chemistry_hamiltonian(E0,K,V,Elt)[1]

function mapped_quantum_chemistry_hamiltonian(E0,K,V,Elt=ComplexF64)
    basis_size = size(K,1);
    half_basis_size = Int(ceil(basis_size/2));
    
    # the phsyical space
    psp = Vect[(Irrep[U₁] ⊠ Irrep[SU₂] ⊠ FermionParity)]((0, 0, 0) => 1,
                                                         (1, 1 // 2, 1) => 1,
                                                         (2, 0, 0) => 1)

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

    ai = isomorphism(storagetype(ap),psp,psp);
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

    indmap_2L = fill(0,2,basis_size,2,basis_size);
    indmap_2R = fill(0,2,basis_size,2,basis_size);
    for pm1 in 1:2, i in 1:half_basis_size, pm2 in 1:2, j in i:half_basis_size
        cnt += 1
        indmap_2L[pm1,i,pm2,j] = cnt;
        indmap_2R[pm1,end-j+1,pm2,end-i+1] = cnt
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
    pp_f = isometry(fuse(_lastspace(ap)' * _lastspace(ap)'),
                    _lastspace(ap)' * _lastspace(ap)')
    mm_f = isometry(fuse(_lastspace(am)' * _lastspace(am)'),
                    _lastspace(am)' * _lastspace(am)')
    mp_f = isometry(fuse(_lastspace(am)' * _lastspace(ap)'),
                    _lastspace(am)' * _lastspace(ap)')
    pm_f = isometry(fuse(_lastspace(ap)' * _lastspace(am)'),
                    _lastspace(ap)' * _lastspace(am)')


    @plansor ut_apap[-1 -2;-3 -4] := ut[-1]*ap[-3 1;3]*ap[1 -2;4]*conj(pp_f[-4;3 4]);
    @plansor ut_amam[-1 -2;-3 -4] := ut[-1]*am[-3 1;3]*am[1 -2;4]*conj(mm_f[-4;3 4]);
    @plansor ut_amap[-1 -2;-3 -4] := ut[-1]*am[-3 1;3]*ap[1 -2;4]*conj(mp_f[-4;3 4]);
    @plansor ut_apam[-1 -2;-3 -4] := ut[-1]*ap[-3 1;3]*am[1 -2;4]*conj(pm_f[-4;3 4]);

    @plansor bpbp_ut[-1 -2;-3 -4] := mm_f[-1;1 2]*bp[1;-3 3]*bp[2;3 -2]*conj(ut[-4]);
    @plansor bmbm_ut[-1 -2;-3 -4] := pp_f[-1;1 2]*bm[1;-3 3]*bm[2;3 -2]*conj(ut[-4]);
    @plansor bmbp_ut[-1 -2;-3 -4] := pm_f[-1;1 2]*bm[1;-3 3]*bp[2;3 -2]*conj(ut[-4]);
    @plansor bpbm_ut[-1 -2;-3 -4] := mp_f[-1;1 2]*bp[1;-3 3]*bm[2;3 -2]*conj(ut[-4]);
    
    for i in 1:basis_size
        if i<half_basis_size
            hamdat[i,1,indmap_2L[1,i,1,i]] += ut_apap;
            hamdat[i,1,indmap_2L[2,i,1,i]] += ut_amap;
            hamdat[i,1,indmap_2L[1,i,2,i]] += ut_apam;
            hamdat[i,1,indmap_2L[2,i,2,i]] += ut_amam;
            for loc in i+1:half_basis_size-1
                hamdat[loc,indmap_2L[1,i,1,i],indmap_2L[1,i,1,i]] = Elt(1);
                hamdat[loc,indmap_2L[2,i,1,i],indmap_2L[2,i,1,i]] = Elt(1);
                hamdat[loc,indmap_2L[1,i,2,i],indmap_2L[1,i,2,i]] = Elt(1);
                hamdat[loc,indmap_2L[2,i,2,i],indmap_2L[2,i,2,i]] = Elt(1);
            end
        elseif i > half_basis_size
            hamdat[i,indmap_2R[1,i,1,i],end] += bpbp_ut;
            hamdat[i,indmap_2R[2,i,1,i],end] += bmbp_ut;
            hamdat[i,indmap_2R[1,i,2,i],end] += bpbm_ut;
            hamdat[i,indmap_2R[2,i,2,i],end] += bmbm_ut;
            for loc in half_basis_size+1:i-1
                hamdat[loc,indmap_2R[1,i,1,i],indmap_2R[1,i,1,i]] = Elt(1);
                hamdat[loc,indmap_2R[2,i,1,i],indmap_2R[2,i,1,i]] = Elt(1);
                hamdat[loc,indmap_2R[1,i,2,i],indmap_2R[1,i,2,i]] = Elt(1);
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
        if j<half_basis_size
            hamdat[j,indmap_1L[1,i],indmap_2L[1,i,1,j]] += p_ap;
            hamdat[j,indmap_1L[1,i],indmap_2L[1,i,2,j]] += p_am;
            hamdat[j,indmap_1L[2,i],indmap_2L[2,i,1,j]] += m_ap;
            hamdat[j,indmap_1L[2,i],indmap_2L[2,i,2,j]] += m_am;
            for k in j+1:half_basis_size-1
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
            for k in half_basis_size+1:i-1
                hamdat[k,indmap_2R[1,i,1,j],indmap_2R[1,i,1,j]] = Elt(1);
                hamdat[k,indmap_2R[1,i,2,j],indmap_2R[1,i,2,j]] = Elt(1);
                hamdat[k,indmap_2R[2,i,2,j],indmap_2R[2,i,2,j]] = Elt(1);
                hamdat[k,indmap_2R[2,i,1,j],indmap_2R[2,i,1,j]] = Elt(1);
            end
        end

    end
    
    
    # fill in all T terms
    # 1 | 1
    for i in 1:basis_size, j in (i + 1):basis_size
        # m | p .
        hamdat[j,map_1[2,i],end] += K[j,i]*bp_ut;
        
        # p | . m
        hamdat[j,map_1[1,i],end] += K[i,j]*bm_ut;
    end
    
    
    # fill in all V terms

    
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
    
    # 3|1
    for i in 1:basis_size,j in i+1:basis_size
        # p p . m | m
        @plansor pp_m[-1 -2;-3 -4] := ut[-1]*ap[1 -2;-4]*h_pm[1;-3]
        hamdat[i,1,map_4[2,j]] += V[i,i,j,i]*pp_m

        # p p m . | m
        @plansor ppm_[-1 -2;-3 -4] := ut[-1]*ap[1 -2;-4]*h_pm[1;-3]
        hamdat[i,1,map_4[2,j]] += V[i,i,i,j]*ppm_
        
        # . p m m | p
        @plansor _pmm[-1 -2;-3 -4] := ut[-1]*h_pm[-2;1]*am[-3 1;-4]
        hamdat[i,1,map_4[1,j]] += V[j,i,i,i]*_pmm

        # p . m m | p
        @plansor p_mm[-1 -2;-3 -4] := ut[-1]*h_pm[-2;1]*am[-3 1;-4]
        hamdat[i,1,map_4[1,j]] += V[i,j,i,i]*p_mm
    end

    # 1|3
    for i in 1:basis_size,j in i+1:basis_size
        # m | p p . m
        @plansor pp_m[-1 -2;-3 -4] := bp[-1;1 -2]*h_pm[1;-3]*conj(ut[-4])
        hamdat[j,map_1[2,i],end] += V[j,j,i,j]*pp_m;

        # m | p p m .
        @plansor ppm_[-1 -2;-3 -4] := bp[-1;1 -2]*h_pm[1;-3]*conj(ut[-4])
        hamdat[j,map_1[2,i],end] += V[j,j,j,i]*ppm_;

        # p | . p m m
        @plansor _pmm[-1 -2;-3 -4] := bm[-1;-3 1]*h_pm[-2;1]*conj(ut[-4])
        hamdat[j,map_1[1,i],end] += V[i,j,j,j]*_pmm;

        # p | p . m m
        @plansor p_mm[-1 -2;-3 -4] := bm[-1;-3 1]*h_pm[-2;1]*conj(ut[-4])
        hamdat[j,map_1[1,i],end] += V[j,i,j,j]*p_mm;
    end
    
    
    # 2|2
    for i in 1:basis_size, j in i+1:basis_size
        # p p | . . m m
        @plansor __mm[-1 -2;-3 -4] := pp_f[-1;1 2]*bm[1;-3 3]*bm[2;3 -2]*conj(ut[-4])
        hamdat[j,map_2[1,i,1,i],end] += V[i,i,j,j]*__mm;

        # m m | p p . .
        @plansor __pp[-1 -2;-3 -4] := mm_f[-1;1 2]*bp[1;-3 3]*bp[2;3 -2]*conj(ut[-4])
        hamdat[j,map_2[2,i,2,i],end] += V[j,j,i,i]*__pp;

        # p m | . p . m
        @plansor _p_m[-1 -2;-3 -4] := mp_f[-1;1 2] * bp[1;-3 3]*bm[2;3 -2]*conj(ut[-4])
        hamdat[j,map_2[2,i,1,i],end] += V[i,j,i,j]*_p_m;
        hamdat[i,1,end] -= V[i,j,i,j]*h_pm;

        # p m | p . m .
        p_m_ = _p_m;
        hamdat[j,map_2[2,i,1,i],end] += V[j,i,j,i]*p_m_;
        hamdat[i,1,end] -= V[j,i,j,i]*h_pm;
        
        # p m | . p m .
        @plansor _pm_[-1 -2;-3 -4] := ut[-1]*h_pm[-2;-3]*conj(ut[-4])
        hamdat[j,map_3[2,i,1,i],end] += V[i,j,j,i]*_pm_;

        # p m | p . . m
        p__m = _pm_;
        hamdat[j,map_3[2,i,1,i],end] += V[j,i,i,j]*p__m;
    end

    # 1|2|1
    for i in 1:basis_size,j in i+1:basis_size,k in j+1:basis_size
        # L R m m
        @plansor LRmm[-1 -2;-3 -4] := am[1 -2;-4]*bm[-1;-3 1]
        hamdat[j,map_1[1,i],map_4[1,k]] += V[i,k,j,j]*LRmm

        # p p L R
        @plansor LRmm[-1 -2;-3 -4] := ap[1 -2;-4]*bp[-1;-3 1]
        hamdat[j,map_1[2,i],map_4[2,k]] += V[j,j,i,k]*LRmm

        # L p R m
        @plansor LpRm[-1 -2;-3 -4] := ap[1 -2;-4]*bm[-1;-3 1]
        hamdat[j,map_1[1,i],map_4[2,k]] += V[i,j,k,j]*LpRm

        # R p L m
        @plansor RpLm[-1 -2;-3 -4] := bp[-1;1 -2]*am[-3 1;-4]
        hamdat[j,map_1[2,i],map_4[1,k]] += V[k,j,i,j]*RpLm

        # p L m R
        @plansor pLmR[-1 -2;-3 -4] := ap[1 -2;-4]*bm[-1;-3 1]
        hamdat[j,map_1[1,i],map_4[2,k]] += V[j,i,j,k]*pLmR

        # L p m R
        @plansor LpmR[-1 -2;-3 -4] := h_pm[4;-3]*iso_pp[-1;3]*τ[3 -2;4 -4]
        hamdat[j,map_1[1,i],map_4[2,k]] += V[i,j,j,k]*LpmR
        
        # p L R m
        @plansor pLRm[-1 -2;-3 -4] := iso_pp[-1;1]*τ[1 -2;2 -4]*h_pm[2;-3]
        hamdat[j,map_1[1,i],map_4[2,k]] += V[j,i,k,j]*pLRm

        # R L m m
        @plansor RLmm[-1 -2;-3 -4] := bm[-1;1 -2]*am[-3 1;-4]
        hamdat[j,map_1[1,i],map_4[1,k]] += V[k,i,j,j]*RLmm;
        
        # p R m L
        @plansor pRmL[-1 -2;-3 -4] := bp[-1;1 -2]*am[-3 1;-4]
        hamdat[j,map_1[2,i],map_4[1,k]] += V[j,k,j,i]*pRmL

        # p p R L
        @plansor ppRL[-1 -2;-3 -4] := bp[-1;1 -2]*ap[-3 1;-4]
        hamdat[j,map_1[2,i],map_4[2,k]] += V[j,j,k,i]*ppRL

        # R p m L
        @plansor RpmL[-1 -2;-3 -4] := h_pm[4;-3]*iso_mm[-1;3]*τ[3 -2;4 -4]
        hamdat[j,map_1[2,i],map_4[1,k]] += V[k,j,j,i]*RpmL

        # p R L m
        @plansor pRLm[-1 -2;-3 -4] := iso_mm[-1;1]*τ[1 -2;2 -4]*h_pm[2;-3]
        hamdat[j,map_1[2,i],map_4[1,k]] += V[j,k,i,j]*pRLm
    end

    # 2|1|1
    # (i,j) in map_2, 1 in map_4, 1 onsite
    for i in 1:basis_size,j in i+1:basis_size,k in j+1:basis_size
        # L L m R
        @plansor LLmR[-1 -2;-3 -4] := pp_f[-1;1 2]*bm[1;-3 3]*τ[3 2;-4 -2]
        hamdat[j,map_2[1,i,1,i],map_4[2,k]] += V[i,i,j,k]*LLmR

        # L L R m
        @plansor LLRm[-1 -2;-3 -4] := pp_f[-1;1 2]*bm[2;3 -2]*τ[1 3;-3 -4]
        hamdat[j,map_2[1,i,1,i],map_4[2,k]] += V[i,i,k,j]*LLRm

        # p L R L
        # L p L R
        @plansor LpLR[-1 -2;-3 -4] := mp_f[-1;1 2]*bp[1;-3 3]*τ[3 2;-4 -2]
        hamdat[j,map_2[2,i,1,i],map_4[2,k]] += V[i,j,i,k]*LpLR

        # L R L m
        @plansor LRLm[-1 -2;-3 -4] := mp_f[-1;1 2]*bm[2;3 -2]*τ[1 3;-3 -4]
        hamdat[j,map_2[2,i,1,i],map_4[1,k]] += V[i,k,i,j]*LRLm

        # p L R L
        @plansor pLRL[-1 -2;-3 -4] := mp_f[-1;1 2]*bp[1;-3 3]*τ[3 2;-4 -2]
        hamdat[j,map_2[2,i,1,i],map_4[2,k]] += V[j,i,k,i]*pLRL

        # R L m L
        @plansor RLmL[-1 -2;-3 -4] := mp_f[-1;1 2]*bm[2;3 -2]*τ[1 3;-3 -4]
        hamdat[j,map_2[2,i,1,i],map_4[1,k]] += V[k,i,j,i]*RLmL

        # R p L L
        @plansor RpLL[-1 -2;-3 -4] := mm_f[-1;1 2]*bp[2;3 -2]*τ[1 3;-3 -4]
        hamdat[j,map_2[2,i,2,i],map_4[1,k]] += V[k,j,i,i]*RpLL

        # p R L L
        @plansor pRLL[-1 -2;-3 -4] := mm_f[-1;1 2]*bp[1;-3 3]*τ[3 2;-4 -2]
        hamdat[j,map_2[2,i,2,i],map_4[1,k]] += V[j,k,i,i]*pRLL

        # L p R L
        @plansor LpRL[-1 -2;-3 -4] := ut[-1]*ap[-3 -2;-4]
        hamdat[j,map_3[2,i,1,i],map_4[2,k]] += V[i,j,k,i]*LpRL

        # L R m L
        @plansor LRpL[-1 -2;-3 -4] := ut[-1]*am[-3 -2;-4]
        hamdat[j,map_3[2,i,1,i],map_4[1,k]] += V[i,k,j,i]*LRpL
        
        # p L L R
        @plansor pLLR[-1 -2;-3 -4] := ut[-1]*ap[-3 -2;-4]
        hamdat[j,map_3[2,i,1,i],map_4[2,k]] += V[j,i,i,k]*pLLR

        # R L L m
        @plansor RLLm[-1 -2;-3 -4] := ut[-1]*am[-3 -2;-4]
        hamdat[j,map_3[2,i,1,i],map_4[1,k]] += V[k,i,i,j]*RLLm
    end
    
    
    # 1|1|2
    # (i,j) in map_2, 2 onsite
    for i in 1:basis_size,j in i+1:basis_size,k in j+1:basis_size
        # j i m m
        @plansor jimm[-1 -2;-3 -4] := bm[1;-3 2]*bm[3;2 -2]*τ[4 5;1 3]*pp_f[-1;4 5]*conj(ut[-4])
        hamdat[k,map_2[1,i,1,j],end] += V[j,i,k,k]*jimm
        # i j m m
        @plansor ijmm[-1 -2;-3 -4] := bm[1;-3 2]*bm[3;2 -2]*τ[4 5;1 3]*permute(pp_f,(1,),(3,2))[-1;4 5]*conj(ut[-4])
        hamdat[k,map_2[1,i,1,j],end] += V[i,j,k,k]*ijmm;

        # j p i m
        @plansor jpim[-1 -2;-3 -4] := bm[1;-3 2]*bp[3;2 -2]*τ[4 5;1 3]*mp_f[-1;4 5]*conj(ut[-4])
        hamdat[k,map_2[2,i,1,j],end] += V[j,k,i,k]*jpim
        # i p j m
        @plansor ipjm[-1 -2;-3 -4] := bm[1;-3 2]*bp[3;2 -2]*τ[4 5;1 3]*permute(pm_f,(1,),(3,2))[-1;4 5]*conj(ut[-4])
        hamdat[k,map_2[1,i,2,j],end] += V[i,k,j,k]*ipjm

        
        # j p m i
        @plansor jpmi[-1 -2;-3 -4] := ut[-1]*h_pm[-2;-3]*conj(ut[-4])
        hamdat[k,map_3[2,i,1,j],end] += V[j,k,k,i]*jpmi
        # i p m j
        @plansor ipmj[-1 -2;-3 -4] := ut[-1]*h_pm[-2;-3]*conj(ut[-4])
        hamdat[k,map_3[1,i,2,j],end] += V[i,k,k,j]*ipmj

        # p p j i
        @plansor ppji[-1 -2;-3 -4] := mm_f[-1;1 2]*bp[1;-3 3]*bp[2;3 -2]*conj(ut[-4])
        hamdat[k,map_2[2,i,2,j],end] += V[k,k,j,i]*ppji
        # p p i j
        @plansor ppij[-1 -2;-3 -4] := permute(mm_f,(1,),(3,2))[-1;1 2]*bp[1;-3 3]*bp[2;3 -2]*conj(ut[-4])
        hamdat[k,map_2[2,i,2,j],end] += V[k,k,i,j]*ppij
        

        # p . . m
        pjim = jpmi;
        hamdat[k,map_3[2,i,1,j],end] += V[k,j,i,k]*pjim
        pijm = ipmj;
        hamdat[k,map_3[1,i,2,j],end] += V[k,i,j,k]*pijm

        # p . m . 
        pjmi = jpim;
        hamdat[k,map_2[2,i,1,j],end] += V[k,j,k,i]*pjmi
        pimj = ipjm;
        hamdat[k,map_2[1,i,2,j],end] += V[k,i,k,j]*pimj
    end
    
    
    # 1|1|1|1
    # (i,j) in map_2, 1 in map_4, 1 onsite

    for i in 1:basis_size,j in i+1:basis_size,k in j+1:basis_size,l in k+1:basis_size
        # p j i R
        @plansor pjiR[-1 -2;-3 -4] := ut[-1]*ap[-3 -2;-4]
        hamdat[k,map_3[2,i,1,j],map_4[2,l]] += V[k,j,i,l]*pjiR
        
        # p i j R
        @plansor pijR[-1 -2;-3 -4] := ut[-1]*ap[-3 -2;-4]
        hamdat[k,map_3[1,i,2,j],map_4[2,l]] += V[k,i,j,l]*pijR

        # R j i m
        @plansor Rjim[-1 -2;-3 -4] := ut[-1]*am[-3 -2;-4]
        hamdat[k,map_3[2,i,1,j],map_4[1,l]] += V[l,j,i,k]*Rjim
        
        # R i j m
        @plansor Rijm[-1 -2;-3 -4] :=ut[-1]*am[-3 -2;-4]
        hamdat[k,map_3[1,i,2,j],map_4[1,l]] += V[l,i,j,k]*Rijm
        
        # j p R i
        @plansor jpRi[-1 -2;-3 -4] := ut[-1]*ap[-3 -2;-4]
        hamdat[k,map_3[2,i,1,j],map_4[2,l]] += V[j,k,l,i]*jpRi

        # i p R j
        @plansor ipRj[-1 -2;-3 -4] := ut[-1]*ap[-3 -2;-4]
        hamdat[k,map_3[1,i,2,j],map_4[2,l]] += V[i,k,l,j]*ipRj

        # j R m i
        @plansor jRmi[-1 -2;-3 -4] := ut[-1]*am[-3 -2;-4]
        hamdat[k,map_3[2,i,1,j],map_4[1,l]] += V[j,l,k,i]*jRmi

        # i R m j
        @plansor iRmj[-1 -2;-3 -4] := ut[-1]*am[-3 -2;-4]
        hamdat[k,map_3[1,i,2,j],map_4[1,l]] += V[i,l,k,j]*iRmj

        # j i m R
        @plansor jimR[-1 -2;-3 -4] := pp_f[-1;1 2]*τ[3 2;-4 -2]*bm[1;-3 3]
        hamdat[k,map_2[1,i,1,j],map_4[2,l]] += V[j,i,k,l]*jimR

        # i j m R
        @plansor ijmR[-1 -2;-3 -4] := permute(pp_f,(1,),(3,2))[-1;1 2]*τ[3 2;-4 -2]*bm[1;-3 3]
        hamdat[k,map_2[1,i,1,j],map_4[2,l]] += V[i,j,k,l]*ijmR

        # j i R m
        @plansor jiRm[-1 -2;-3 -4] := pp_f[-1;1 2]*τ[1 3;-3 -4]*bm[2;3 -2]
        hamdat[k,map_2[1,i,1,j],map_4[2,l]] += V[j,i,l,k]*jiRm

        # i j R m
        @plansor ijRm[-1 -2;-3 -4] := permute(pp_f,(1,),(3,2))[-1;1 2]*τ[1 3;-3 -4]*bm[2;3 -2]
        hamdat[k,map_2[1,i,1,j],map_4[2,l]] += V[i,j,l,k]*ijRm

        # j p i R
        @plansor jpiR[-1 -2;-3 -4] := mp_f[-1;1 2]*τ[3 2;-4 -2]*bp[1;-3 3]
        hamdat[k,map_2[2,i,1,j],map_4[2,l]] += V[j,k,i,l]*jpiR

        # i p j R
        @plansor ipjR[-1 -2;-3 -4] := permute(pm_f,(1,),(3,2))[-1;1 2]*τ[3 2;-4 -2]*bp[1;-3 3]
        hamdat[k,map_2[1,i,2,j],map_4[2,l]] += V[i,k,j,l]*ipjR

        # j R i m
        @plansor jRim[-1 -2;-3 -4] := mp_f[-1;1 2]*τ[1 3;-3 -4]*bm[2;3 -2]
        hamdat[k,map_2[2,i,1,j],map_4[1,l]] += V[j,l,i,k] * jRim

        # i R j m
        @plansor iRjm[-1 -2;-3 -4] := permute(pm_f,(1,),(3,2))[-1;1 2]*τ[1 3;-3 -4]*bm[2;3 -2]
        hamdat[k,map_2[1,i,2,j],map_4[1,l]] += V[i,l,j,k] * iRjm
        
        # p j R i
        @plansor pjRi[-1 -2;-3 -4] := mp_f[-1;1 2]*τ[3 2;-4 -2]*bp[1;-3 3];
        hamdat[k,map_2[2,i,1,j],map_4[2,l]] += V[k,j,l,i] * pjRi

        # p i R j
        @plansor piRj[-1 -2;-3 -4] := permute(pm_f,(1,),(3,2))[-1;1 2]*τ[3 2;-4 -2]*bp[1;-3 3];
        hamdat[k,map_2[1,i,2,j],map_4[2,l]] += V[k,i,l,j] * piRj

        # R j m i
        @plansor Rjmi[-1 -2;-3 -4] := mp_f[-1;1 2]*τ[1 3;-3 -4]*bm[2;3 -2]
        hamdat[k,map_2[2,i,1,j],map_4[1,l]] += V[l,j,k,i]*Rjmi

        # R i m j
        @plansor Rimj[-1 -2;-3 -4] := permute(pm_f,(1,),(3,2))[-1;1 2]*τ[1 3;-3 -4]*bm[2;3 -2]
        hamdat[k,map_2[1,i,2,j],map_4[1,l]] += V[l,i,k,j]*Rimj

        # R p j i
        @plansor Rpji[-1 -2;-3 -4] := mm_f[-1;1 2]*τ[1 3;-3 -4]*bp[2;3 -2]
        hamdat[k,map_2[2,i,2,j],map_4[1,l]] += V[l,k,j,i]*Rpji

        # R p i j
        @plansor Rpij[-1 -2;-3 -4] := permute(mm_f,(1,),(3,2))[-1;1 2]*τ[1 3;-3 -4]*bp[2;3 -2]
        hamdat[k,map_2[2,i,2,j],map_4[1,l]] += V[l,k,i,j]*Rpij

        # p R j i
        @plansor pRji[-1 -2;-3 -4] := mm_f[-1;1 2]*τ[3 2;-4 -2]*bp[1;-3 3]
        hamdat[k,map_2[2,i,2,j],map_4[1,l]] += V[k,l,j,i]*pRji

        # p R i j
        @plansor pRij[-1 -2;-3 -4] := permute(mm_f,(1,),(3,2))[-1;1 2]*τ[3 2;-4 -2]*bp[1;-3 3]
        hamdat[k,map_2[2,i,2,j],map_4[1,l]] += V[k,l,i,j]*pRij
    end

    th = MPOHamiltonian(map(x->x == InitialValue(+) ? missing : x,hamdat));
    #th = MPOHamiltonian(MPSKit.remove_orphans(th.data));
    th+fill(E0/basis_size,basis_size),(map_1,map_2,map_3,map_4)
end