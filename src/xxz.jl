function nonsym_xxz_ham(;spin = 1,delta = 1,zfield = 0.0)
    (sx,sy,sz,id) = nonsym_spintensors(spin)

    mpo=Array{typeof(sx),3}(undef,1,5,5)
    mpo[1,:,:]=[id sx sy delta*sz zfield*sz;
                0*id 0*id 0*id 0*id sx;
                0*id 0*id 0*id 0*id sy;
                0*id 0*id 0*id 0*id sz;
                0*id 0*id 0*id 0*id id]

    return MPOHamiltonian(mpo)
end

function su2_xxx_ham(;spin = 1//2)
    #only checked for spin = 1 and spin = 2...
    ph = Rep[SU₂](spin=>1)

    Sl1 = TensorMap(ones, ComplexF64, ph , Rep[SU₂](1=>1)*ph)*sqrt(spin^2+spin)
    Sr1 = TensorMap(ones, ComplexF64, Rep[SU₂](1=>1)*ph ,ph)*sqrt(spin^2+spin)

    @tensor NN[-1 -2;-3 -4] := Sl1[-1;2 -3]*Sr1[2 -2;-4]

    return MPOHamiltonian(NN);
end

function u1_xxz_ham(;spin = 1,delta = 1,zfield = 0.0)
    (sxd,syd,szd,idd) = spinmatrices(spin);
    @tensor ham[-1 -2;-3 -4]:=sxd[-1,-3]*sxd[-2,-4]+syd[-1,-3]*syd[-2,-4]+(delta*szd)[-1,-3]*szd[-2,-4]+zfield*0.5*szd[-1,-3]*idd[-2,-4]+zfield*0.5*idd[-1,-3]*szd[-2,-4]

    indu1map = [Irrep[U₁](v) for v in -spin:1:spin];
    pspace = U1Space((v=>1 for v in indu1map));

    symham = TensorMap(zeros,eltype(ham),pspace*pspace,pspace*pspace)

    for i in 1:size(ham,1),
        j in 1:size(ham,1),
        k in 1:size(ham,1),
        l in 1:size(ham,1)
        if ham[i,j,k,l]!=0
            copy!(symham[(indu1map[i],indu1map[j],indu1map[end-k+1],indu1map[end-l+1])],ham[i:i,j:j,k:k,l:l])
        end
    end

    return MPOHamiltonian(MPSKit.decompose_localmpo(MPSKit.add_util_leg(symham)))
end
