"""
a quantum link model of qed
"""
function nonsym_qed_qlm_ham(;m=0.8,a=1,e=1.0,l=1)
    (Xl,Yl,Zl,Il) = nonsym_spintensors(l);
    Pl = Xl + 1im*Yl; Ml = Xl - 1im*Yl;
    (Xf,Yf,Zf,If) = nonsym_spintensors(1//2);
    Pf = Xf + 1im*Yf; Mf = Xf - 1im*Yf;

    data = Array{Union{Missing,typeof(Pf)},3}(missing,2,4,4)

    data[1,1,1] = Il; data[1,end,end] = Il;
    data[2,1,1] = If; data[2,end,end] = If;

    data[1,1,end] = a*e*e*0.5*Zl*Zl
    data[2,1,end] = m*Zf

    pref = -1im/(a*2*sqrt(l*(l+1)))

    data[2,1,2] = pref*Pf
    data[1,2,2] = Pl
    data[2,2,end] = Pf

    data[2,1,3] = -pref*Mf
    data[1,3,3] = Ml
    data[2,3,end] = Mf

    return MPOHamiltonian(data)
end

"""
qed_qlm has a local gauge transform G_j, here we construct G_j^2
useful to find the groundstate of qed_qlm in the G = 0 sector
"""
function qed_qlm_G2(;l=1)
    (Xl,Yl,Zl,Il) = nonsym_spintensors(l);
    Pl = Xl + 1im*Yl; Ml = Xl - 1im*Yl;

    (Xf,Yf,Zf,If) = nonsym_spintensors(1//2);
    Pf = Xf + 1im*Yf; Mf = Xf - 1im*Yf;

    @tensor todecomp[-1 -2 -3;-4 -5 -6]:= Zl[-1;-4]*If[-2;-5]*Il[-3;-6]
    @tensor todecomp[-1 -2 -3;-4 -5 -6]+= Il[-1;-4]*If[-2;-5]*Zl[-3;-6]
    @tensor todecomp[-1 -2 -3;-4 -5 -6]-= Il[-1;-4]*(Zf+If*0.5)[-2;-5]*Il[-3;-6]
    todecomp = todecomp*todecomp

    string = MPSKit.decompose_localmpo(add_util_leg(todecomp))


    nIl = add_util_leg(Il);
    nIf = add_util_leg(If);

    data = Array{Union{Missing,typeof(string[1])},3}(missing,2,3,3)

    data[1,1,1] = nIl; data[1,end,end] = nIl;
    data[2,1,1] = nIf; data[2,end,end] = nIf;

    data[1,1,2] = string[1];
    data[2,2,2] = string[2];
    data[1,2,end] = string[3];

    return MPOHamiltonian(data)
end


"""
the lattice qed hamiltonian, using a seperate U(1) charge for even and odd matter sites, allowing one to construct manifestly guage invariant mps's
"""
function u1_qed_ham(;m=1,g=2,max_U=3)
    # 2 U_1 charges, even and odd
    link_space = Rep[U₁×U₁]((i,-i)=>1 for i in -max_U:max_U);
    odd_matter_space = Rep[U₁×U₁]((0,0)=>1,(0,1)=>1);
    even_matter_space = Rep[U₁×U₁]((-1,0)=>1,(0,0)=>1);

    #the E operator is gauge invariant, so we can make it
    E = TensorMap(ComplexF64,link_space,link_space)
    for (f1,f2) in fusiontrees(E) #there are easier ways than iterating over the fusiontrees, but I'm not sure how
        curent_charge = f1.coupled.sectors[1].charge;
        E[f1,f2] = fill(curent_charge,1,1)
    end

    #even_matter_N
    even_matter_N = TensorMap(ComplexF64,even_matter_space,even_matter_space)
    for (f1,f2) in fusiontrees(even_matter_N)
        curent_charge = f1.coupled.sectors[1].charge;
        even_matter_N[f1,f2] = fill(curent_charge == 0 ? 0 : 1,1,1);
    end

    #odd_matter_N
    odd_matter_N = TensorMap(ComplexF64,odd_matter_space,odd_matter_space)
    for (f1,f2) in fusiontrees(odd_matter_N)
        curent_charge = f1.coupled.sectors[2].charge;
        odd_matter_N[f1,f2] = fill(curent_charge == 1 ? 0 : 1,1,1);
    end

    #the U operator is not gauge invariant, so it has to also act on the virtual level!
    virtual_link_space = Rep[U₁×U₁]((0,1)=>1,(0,-1)=>1,(1,0)=>1,(-1,0)=>1);
    ou = oneunit(virtual_link_space);
    U = TensorMap(ones,ComplexF64,virtual_link_space*link_space,link_space*virtual_link_space);
    #I think - by construction - that U will act like a creation/anihilation operator

    #odd_matter_flip
    odd_matter_flip_1 = TensorMap(ones,ComplexF64,ou*odd_matter_space,odd_matter_space*virtual_link_space)
    odd_matter_flip_2 = TensorMap(ones,ComplexF64,virtual_link_space*odd_matter_space,odd_matter_space*ou)

    even_matter_flip_1 = TensorMap(ones,ComplexF64,ou*even_matter_space,even_matter_space*virtual_link_space)
    even_matter_flip_2 = TensorMap(ones,ComplexF64,virtual_link_space*even_matter_space,even_matter_space*ou)

    #think this fixes jordan wigner?
    blocks(odd_matter_flip_1)[U₁(0)⊠U₁(1)] .*= -1;
    blocks(even_matter_flip_1)[U₁(0)⊠U₁(0)] .*= -1;

    #matter - even - matter - odd
    data = Array{Any,3}(missing,4,3,3);
    data[:,1,1] .= 1.0
    data[:,3,3] .= 1.0

    #even matter site:
    data[1,1,2] = even_matter_flip_1;
    data[1,2,3] = even_matter_flip_2;
    data[1,1,3] = m * even_matter_N;

    #link
    data[2,1,3] = g^2/2 * E^2
    data[2,2,2] = U * 1im/2;

    #odd matter site
    data[3,1,2] = odd_matter_flip_1;
    data[3,2,3] = odd_matter_flip_2;
    data[3,1,3] = -m * odd_matter_N;

    #link
    data[4,1,3] = g^2/2 * E^2;
    data[4,2,2] = U* 1im/2;

    MPOHamiltonian(data)
end
