using ADIPoisson, Plots, PiecewiseOrthogonalPolynomials, LaTeXStrings, DelimitedFiles
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


n = 9
r = range(-1,1; length=n+1)
p = 20

δt = 0.001

(x,y), pl_P,pl_Q = plan_poisson_transform(n, p);
L⁻¹ = plan_poissonsolve!(n, p; ω = 1/δt);
# (x,y),L⁻¹,pl_P, pl_Q = plan_poissonsolve!(n, p, ω = 1/δt);  #; ω = 10^2);


F = f.(x, reshape(y,1,1,size(y)...));
U_0 = pl_P * F

U = [U_0]

for _ = 1:10
    U_k = last(U)
    V_k = pl_Q \ (L⁻¹ * (U_k/δt));
    V_k .= V_k .- δt .* V_k.^2;
    push!(U, pl_P * V_k)
end

contourf(vec(x[end:-1:1,:]), vec(y[end:-1:1,:]), reshape((pl_P \ U[end])[end:-1:1,:,end:-1:1,:], length(x), length(y))')

###
# Burgers
###


# u_t + uu_x = ν*Δu
# discretise by splitting:
# solve u_t = -uu_x explicitly via forward Euler
# u_{k+1} = u_k - h u_kx u_k
# and u_t = Δu implicitly via backward Euler
# (I/(ν*h) - Δ) u_{k+1} = u_k/(ν*h)


n = 9
r = range(-1,1; length=n+1)
p = 20

P = ContinuousPolynomial{0}(r)
Q = DirichletPolynomial(r)
R = (P \ Q)[Block.(oneto(p+1)),Block.(oneto(p))]
D_x = (P\diff(Q))[Block.(oneto(p+1)),Block.(oneto(p))]

ν = 0.01

δt = 0.01

# (x,y),L⁻¹,pl_P, pl_Q = plan_poissonsolve!(n, p, ω = 1/(δt*ν));  #; ω = 10^2);
(x,y), pl_P,pl_Q = plan_poisson_transform(n, p);
L⁻¹ = plan_poissonsolve!(n, p; ω = 1/(δt*ν));


F = f.(x, reshape(y,1,1,size(y)...));
U_0 = pl_P * F

U = [U_0]

function burgers_timestep(U_k, ν, δt,L⁻¹,pl_P, pl_Q, D_x, R)
    W_k = (L⁻¹ * (U_k/(δt*ν))); # heat
    V_kx = pl_P\(D_x*W_k*R');
    V_k = pl_Q\(W_k);
    V_k .= V_k .- δt .* V_kx .* V_k;
    pl_P * V_k
end


for _ = 1:40
    @time push!(U, burgers_timestep(U[end], ν, δt,L⁻¹,pl_P, pl_Q, D_x, R))
end



contourf(vec(x[end:-1:1,:]), vec(y[end:-1:1,:]), reshape((pl_P \ U[51])[end:-1:1,:,end:-1:1,:], length(x), length(y))', linewidth=0, title="Burger's solution (t = 0.5)", nlevels=150)
savefig("burgers_sol.pdf")


# surface(vec(x[end:-1:1,:]), vec(y[end:-1:1,:]), reshape((pl_Q \ W_k)[end:-1:1,:,end:-1:1,:], length(x), length(y))')

# surface(vec(x[end:-1:1,:]), vec(y[end:-1:1,:]), reshape((V_kx)[end:-1:1,:,end:-1:1,:], length(x), length(y))')


###
# timing
###



ν = 0.01

δt = 0.01
n
(x,y),L⁻¹,pl_P, pl_Q = plan_poissonsolve!(n, p, ω = 1/(δt*ν));  #; ω = 10^2);

pps = n -> 2 .^ (2:round(Int, log2(5_000 / n)))


ns = [9, 18]

plt = plot(;title="Burger's Timestep", legend=:bottomright, xscale=:log10, yscale=:log10, xlabel=L"N^2" * " (Total Degrees of Freedom)", ylabel="time (seconds)", xticks=10 .^ (1:8), yticks = 10.0 .^ (-5:3), ylims=[1E-3,1E3])
markers = [:circle, :rect, :diamond, :uptriangle]
m = 0
for n in ns
    m += 1
    @show n
    r = range(-1, 1; length=n+1)
    Q = DirichletPolynomial(r)
    dat = readdlm("data/burgers_n_$n.csv")
    scatter!([size(Q[:,Block.(Base.oneto(p))],2)^2 for p in pps(n)], dat[:,2]; label="n = $n", markershape=markers[m])
end


Ns = 9pps(9); plot!(Ns .^ 2, 4E-7*Ns.^2 .* log.(Ns).^2; label=false, style=:dash)
annotate!(1E6, 3E2, L"O(N^2 log^2 N)")
savefig("burgers_tms.pdf")
