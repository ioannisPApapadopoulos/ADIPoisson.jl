using ClassicalOrthogonalPolynomials, PiecewiseOrthogonalPolynomials, ADIPoisson, Plots
using ClassicalOrthogonalPolynomials: plan_grid_transform, pad


n = 9
r = range(-1,1; length=n+1)
p = 100

P = ContinuousPolynomial{0}(r)

###
# HP
###
let pix = [tuple.(2, 4:8); (3,6); tuple.(4, 4:6); tuple.(6, 2:6); (7,4); (7,6); tuple.(8, 4:6)]
    global f = function(x,y)
        for (k,j) in pix
            r[k] ≤ x ≤ r[k+1] && r[j] ≤ y ≤ r[j+1] && return 1.0
        end
        return 0.0
    end
end

@time (x,y), pl_P,pl_Q = plan_poisson_transform(n, p);
F = f.(x, reshape(y,1,1,size(y)...));
surface(vec(x[end:-1:1,:]), vec(y[end:-1:1,:]), reshape(F[end:-1:1,:,end:-1:1,:], length(x), length(y))')
savefig("hprhs.pdf")

@time Δ⁻¹ = plan_poissonsolve!(n, p; ω = 10^2);
@time U = pl_Q \ (Δ⁻¹ * (pl_P * F));
surface(vec(x[end:-1:1,:]), vec(y[end:-1:1,:]), reshape(U[end:-1:1,:,end:-1:1,:], length(x), length(y))')
savefig("hpsol.pdf")


ω = 0
errs = Dict()
@time (x,y), pl_P,pl_Q = plan_poisson_transform(n, p);
@time Δ⁻¹ = plan_poissonsolve!(n, p; ω = 10^2);
@time U_ex = (Δ⁻¹ * (pl_P * F));

ps = 10:10:50
for p in ps
    @show p
    (x,y),pl_P,pl_Q = plan_poisson_transform(n, p);
    Δ⁻¹ = plan_poissonsolve!(n, p; ω = ω);
    F = f.(x, reshape(y,1,1,size(y)...));
    errs[(ω,p)] = norm(Matrix(pad((Δ⁻¹ * (pl_P * F)), size(U_ex)...) - U_ex))
end
nanabs = x -> iszero(x) ? NaN : abs(x)
plot(ps, nanabs.(getindex.(Ref(errs), tuple.(ω,ps))); yscale=:log10, xscale=:log10, label = "ω = $ω", xticks=10.0 .^ (0:0.25:2), yticks=10.0 .^ (-5:0.25:-1))


###
# neumann
###


@time (x,y), pl_P,pl_Q = plan_poisson_transform_neumann(n, p);
@time Δ⁻¹ = plan_poissonsolve_neumann!(n, p; ω = 2^2);

F = f.(x, reshape(y,1,1,size(y)...));
surface(vec(x[end:-1:1,:]), vec(y[end:-1:1,:]), reshape(F[end:-1:1,:,end:-1:1,:], length(x), length(y))')
savefig("hprhs.pdf")


U = pl_Q \ (Δ⁻¹ * (pl_P * F));
surface(vec(x[end:-1:1,:]), vec(y[end:-1:1,:]), reshape(U[end:-1:1,:,end:-1:1,:], length(x), length(y))')
savefig("hpsol_neumann.pdf")



###
# FU
###

let pix = [tuple.(2, 2:5); (3,3); (3,5); tuple.(5, 2:5); (6,2); tuple.(7, 2:5)]
    global f = function(x,y)
        for (k,j) in pix
            r[k] ≤ x ≤ r[k+1] && r[j] ≤ y ≤ r[j+1] && return 1.0
        end
        return 0.0
    end
end

F = f.(x, reshape(y,1,1,size(y)...));
contourf(vec(x[end:-1:1,:]), vec(y[end:-1:1,:]), reshape(F[end:-1:1,:,end:-1:1,:], length(x), length(y))')   



Q = DirichletPolynomial(r);
(x,y),pl = plan_poissonsolve!(n, p);
pl_Q = plan_transform(Q, Block(p,p));
U = pl_Q\(pl*F);

contourf(vec(x[end:-1:1,:]), vec(y[end:-1:1,:]), reshape(U[end:-1:1,:,end:-1:1,:], length(x), length(y))')   



###
# Good Helmholtz
(x,y),pl = plan_poissonsolve!(n, p, ω = 10^2);

U = pl_Q\(pl*F);
surface(vec(x[end:-1:1,:]), vec(y[end:-1:1,:]), reshape(U[end:-1:1,:,end:-1:1,:], length(x), length(y))')