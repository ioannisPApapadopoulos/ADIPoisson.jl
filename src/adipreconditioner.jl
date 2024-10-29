struct ADILaplacePreconditioner{Ap,Qp,Pp,Cx,XY,AA,MM,QPP}
    pl_adi::Ap
    pl_Q::Qp
    pl_P::Pp
    coeffx::Cx
    xy::XY
    A::AA
    M::MM
    QP::QPP
    p::Int
    n::Int
end

function ADILaplacePreconditioner(a::Function, r::AbstractVector{T}, p::Int; adi_tolerance::T=1e-14) where T
    n = length(r)-1
    P = ContinuousPolynomial{0}(r)
    Q = DirichletPolynomial(r)

    (x,y),pl_Q = plan_grid_transform(Q, Block(p,p))
    pl_P = plan_transform(P, Block(p+1,p+1))
    coeffx = a.(x,reshape(y,1,1,size(y)...))
    @assert isempty(findall(abs.(coeffx) .> 1e15))

    A = -weaklaplacian(Q)[Block.(oneto(p)), Block.(oneto(p))]
    M = grammatrix(Q)[Block.(oneto(p)), Block.(oneto(p))]
    QP = (Q'*P)[Block.(1:p), Block.(1:p+1)]
    pl_adi = plan_adi!(M, -M, A; factorize = reversecholesky ∘ Symmetric, tolerance=adi_tolerance);
    ADILaplacePreconditioner(pl_adi, pl_Q, pl_P, coeffx, (x,y), A, M, QP, p, n)
end

function apply_ncc(P::ADILaplacePreconditioner, x::AbstractVector{T}) where T
    n, p = P.n, P.p
    A, M, QP = P.A, P.M, P.QP
    pl_Q, pl_P = P.pl_Q, P.pl_P

    X = BlockMatrix(reshape(x, p*n-1, p*n-1), [n-1;repeat([n], p-1)], [n-1;repeat([n], p-1)])
    Y = pl_P * (P.coeffx .* (pl_Q \ X))
    vec( A*X*M' + M*X*A + QP*Y*QP' )
end

function ldiv!(a::AbstractVector{T}, P::ADILaplacePreconditioner, b::AbstractVector{T}) where T
    n, p = P.n, P.p
    X = reshape(b, p*n-1, p*n-1)
    a .= vec(P.pl_adi * copy(X))
end
ldiv!(P::ADILaplacePreconditioner, b::AbstractVector{T}) where T = ldiv!(b,P,b)