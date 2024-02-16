using ADIPoisson, PiecewiseOrthogonalPolynomials, DelimitedFiles
using Base:oneto


# Solve
# u_t = Δu - u^2
# discretise by splitting:
# solve u_t = -u^2 explicitly via forward Euler
# u_{k+1} = u_k - h u_k^2
# and u_t = Δu implicitly via backward Euler
# (I/h - Δ) u_{k+1} = u_k/h



let r = range(-1,1; length=10),
    pix = [tuple.(2, 4:8); (3,6); tuple.(4, 4:6); tuple.(6, 2:6); (7,4); (7,6); tuple.(8, 4:6)]
    global f = function(x,y)
        for (k,j) in pix
            r[k] ≤ x ≤ r[k+1] && r[j] ≤ y ≤ r[j+1] && return 1.0
        end
        return 0.0
    end
end



function burgers_timestep(U_k, ν, δt,L⁻¹,pl_P, pl_Q, D_x, R)
    W_k = (L⁻¹ * (U_k/(δt*ν))); # heat
    V_kx = pl_P\(D_x*W_k*R');
    V_k = pl_Q\(W_k);
    V_k .= V_k .- δt .* V_kx .* V_k;
    pl_P * V_k
end


###
# timing
###



ν = 0.01
δt = 0.01
#compile
let n = 9, p = 10
    L⁻¹ = plan_poissonsolve!(n, p, ω = 1/(δt*ν));
    (x,y),pl_P, pl_Q = plan_poisson_transform(n, p);
    F = f.(x, reshape(y,1,1,size(y)...));
    @time U_0 = pl_P * F
    r = range(-1, 1; length=n+1)
    P = ContinuousPolynomial{0}(r)
    Q = DirichletPolynomial(r)
    R = (P \ Q)[Block.(oneto(p+1)),Block.(oneto(p))]
    D_x = (P\diff(Q))[Block.(oneto(p+1)),Block.(oneto(p))]

    @time burgers_timestep(U_0, ν, δt,L⁻¹,pl_P, pl_Q, D_x, R)
end

timesteptms = Dict()

ns = [9, 18]
pps = n -> 2 .^ (2:round(Int, log2(5_000 / n)))


for n in ns, p in pps(n) # truncation degree on each cell
    @show (n,p)
    L⁻¹ = plan_poissonsolve!(n, p, ω = 1/(δt*ν));
    (x,y),pl_P, pl_Q = plan_poisson_transform(n, p);
    F = f.(x, reshape(y,1,1,size(y)...));
    @time U_0 = pl_P * F
    r = range(-1, 1; length=n+1)
    P = ContinuousPolynomial{0}(r)
    Q = DirichletPolynomial(r)
    R = (P \ Q)[Block.(oneto(p+1)),Block.(oneto(p))]
    D_x = (P\diff(Q))[Block.(oneto(p+1)),Block.(oneto(p))]

    timesteptms[(n,p)] = @elapsed(burgers_timestep(U_0, ν, δt,L⁻¹,pl_P, pl_Q, D_x, R))
end

for n in ns
    ps = pps(n)
    data = [ps getindex.(Ref(timesteptms),tuple.(n,ps))]
    writedlm("data/burgers_n_$n.csv", data)
end
