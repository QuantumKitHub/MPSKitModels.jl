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

    @tensor todecomp[-1 -2 -3;-4 -5 -6]:= Zl[-1 -4]*If[-2 -5]*Il[-3 -6]
    @tensor todecomp[-1 -2 -3;-4 -5 -6]+= Il[-1 -4]*If[-2 -5]*Zl[-3 -6]
    @tensor todecomp[-1 -2 -3;-4 -5 -6]-= Il[-1 -4]*(Zf+If*0.5)[-2 -5]*Il[-3 -6]
    todecomp = todecomp*todecomp

    string = MPSKit.decompose_localmpo(add_util_leg(todecomp))


    nIl = permute(add_util_leg(Il),(1,2),(4,3))
    nIf = permute(add_util_leg(If),(1,2),(4,3))

    data = Array{Union{Missing,typeof(string[1])},3}(missing,2,3,3)

    data[1,1,1] = nIl; data[1,end,end] = nIl;
    data[2,1,1] = nIf; data[2,end,end] = nIf;

    data[1,1,2] = string[1];
    data[2,2,2] = string[2];
    data[1,2,end] = string[3];

    return MPOHamiltonian(data)
end
