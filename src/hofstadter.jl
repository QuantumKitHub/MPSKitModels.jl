function U1_strip_harper_hofstadter(width;flux=pi/2,Jx=1,Jy=1,periodic=false,filling=1)
    #we change the charge of the first site (c - filling), imposing a filling/width
    ps1 = Rep[U₁](-filling=>1,(1-filling)=>1);
    ps2 = Rep[U₁](0=>1,1=>1);

    ou = oneunit(ps1);

    #pspaces[i] = physical space at location i
    pspaces = [ps1;fill(ps2,width-1)];

    hop_y_1 = map(pspaces) do ps
        TensorMap(ones,ComplexF64,ou*ps,ps*Rep[U₁](1=>1,-1=>1));
    end
    hop_y_2 = map(pspaces) do ps
        TensorMap(ones,ComplexF64,Rep[U₁](1=>1,-1=>1)*ps,ps*ou);
    end

    hop_x_1 = map(enumerate(hop_y_1)) do (y,h)
        nh = copy(h);

        upcharge = (y == 1) ? U₁(1-filling) : U₁(1);
        downcharge = (y == 1) ? U₁(-filling) : U₁(0);

        blocks(nh)[upcharge].*=exp(1im*y*flux)
        blocks(nh)[downcharge].*=exp(-1im*y*flux)

        nh
    end
    hop_x_2 = hop_y_2;

    passthrough = map(pspaces) do ps
        complex(isomorphism(Rep[U₁](1=>1,-1=>1)*ps,ps*Rep[U₁](1=>1,-1=>1)));
    end

    #the actual mpo - hamiltonian
    mpot = Array{Any,3}(missing,width,width+3+periodic,width+3+periodic)

    for y in 1:width
        mpot[y,1,1] = 1.0;
        mpot[y,end,end] = 1.0;

        #pass the previously created - anihilated particle through
        for cy in [1:y-1;y+1:width]
            mpot[y,cy+1,cy+1] = passthrough[y]
        end

        #create/anihilate particles along the x direction
        mpot[y,1,y+1] = -hop_x_1[y]*Jx;
        mpot[y,y+1,end] = hop_x_2[y];

        #create/anihilated along the y direction
        mpot[y,1,2+width] = -hop_y_1[y]*Jy

        if y!=1
            mpot[y,2+width,end] = hop_y_2[y];
        end


        #if it's periodic we need to create/anihilate on the endpoints as well
        if periodic
            if y == 1
                mpot[y,1,3+width] = - hop_y_1[y]*Jy
            elseif y == width
                mpot[y,3+width,end] = hop_y_2[y];
            else
                mpot[y,3+width,3+width] = passthrough[y]
            end
        end

    end

    MPOHamiltonian(mpot)

end
