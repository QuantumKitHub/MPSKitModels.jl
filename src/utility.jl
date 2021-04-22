"
    Returns spin operators Sx,Sy,Sz,Id for spin s
"
function spinmatrices(s::Union{Rational{Int},Int},elt = ComplexF64)
    N = Int(2*s)

    Sx = zeros(elt,N+1,N+1)
    Sy = zeros(elt,N+1,N+1)
    Sz = zeros(elt,N+1,N+1)

    for row=1:(N+1)
        for col=1:(N+1)
            term=sqrt((s+1)*(row+col-1)-row*col)/2.0

            if (row+1==col)
                Sx[row,col]+=term
                Sy[row,col]-=1im*term
            end

            if(row==col+1)
                Sx[row,col]+=term
                Sy[row,col]+=1im*term
            end

            if(row==col)
                Sz[row,col]+=s+1-row
            end

        end
    end
    return Sx,Sy,Sz,one(Sx)
end

function nonsym_spintensors(s)
    (Sxd,Syd,Szd) = spinmatrices(s)
    sp = ComplexSpace(size(Sxd,1))

    Sx = TensorMap(Sxd,sp,sp);
    Sy = TensorMap(Syd,sp,sp);
    Sz = TensorMap(Szd,sp,sp);

    return Sx,Sy,Sz,one(Sx)
end

"""
bosonic creation anihilation operators with a cutoff
cutoff = maximal number of bosons at one location
"""
function nonsym_bosonictensors(cutoff::Int,elt=ComplexF64)
    creadat = zeros(elt,cutoff+1,cutoff+1);

    for i in 1:cutoff
        creadat[i+1,i] = sqrt(i);
    end

    a⁺ = TensorMap(creadat,ℂ^(cutoff+1),ℂ^(cutoff+1));
    a⁻ = TensorMap(collect(creadat'),ℂ^(cutoff+1),ℂ^(cutoff+1));
    return (a⁺,a⁻)
end
