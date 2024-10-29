module ADIPoisson

using PiecewiseOrthogonalPolynomials, ClassicalOrthogonalPolynomials, AlternatingDirectionImplicit, BlockArrays, MatrixFactorizations, StaticArrays, LinearAlgebra
import Base: oneto, *
import LinearAlgebra: ldiv!
using ClassicalOrthogonalPolynomials: plan_grid_transform

export plan_poissonsolve!, plan_poissonsolve_neumann!, plan_poisson_transform, plan_poisson_transform_neumann,
        ADILaplacePreconditioner, apply_ncc


include("adipreconditioner.jl")

plan_poissonsolve!(n, p, (a,b) = (-1,1); kwds...) = _plan_poissonsolve!(DirichletPolynomial(range(a, b, n+1)), n, p; kwds...)
plan_poissonsolve_neumann!(n, p, (a,b) = (-1,1); kwds...) = _plan_poissonsolve!(ContinuousPolynomial{1}(range(a, b, n+1)), n, p; kwds...)

function _plan_poissonsolve!(Q, n, p; tolerance=1E-14, ω = 0)
    r = Q.points
    P = ContinuousPolynomial{0}(r)
    Δ = -weaklaplacian(Q)
    M = grammatrix(Q)
    KR = Block.(oneto(p))
    Mₙ = M[KR,KR]
    Δₙ = Δ[KR,KR]
    C = iszero(ω) ? Δₙ : (Δₙ + (ω/2)*Mₙ)

    rc = reversecholesky ∘ Symmetric
    Pl = plan_adi!(Mₙ, -Mₙ, C; factorize = rc, tolerance=tolerance)
    PoissonPlan(Pl, (Q'*P)[KR, Block.(oneto(p+1))])
end

plan_poisson_transform(n, p, (a,b) = (-1,1)) = _plan_poisson_transform(DirichletPolynomial(range(a, b, n+1)), p)
plan_poisson_transform_neumann(n, p, (a,b) = (-1,1)) = _plan_poisson_transform(ContinuousPolynomial{1}(range(a, b, n+1)), p)

function _plan_poisson_transform(Q, p)
    r = Q.points
    P = ContinuousPolynomial{0}(r)
    (x,y),Tr = plan_grid_transform(P, Block(p+1,p+1))
    (x,y),Tr,plan_transform(Q, Block(p,p))
end




struct PoissonPlan{Pl,QPMat}
    padi::Pl
    QP::QPMat
end


function *(pl::PoissonPlan, cfs::AbstractMatrix)
    F = pl.QP*cfs*pl.QP'
    X = pl.padi*F
    BlockedArray(X, (axes(pl.QP,1), axes(pl.QP,1)))
end


end # module ADIPoisson
