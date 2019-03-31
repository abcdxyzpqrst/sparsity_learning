using LinearAlgebra
using Distributions

function objective(X, y, θ, λ, γ)
    s = 0.0                         # fusion penalty 
    for j = 2:length(θ)
        s += abs(θ[j] - θ[j-1])
    end
    return 0.5*mean(abs2, X*θ - y) + λ*sum(abs, θ) + γ*s
end

function lasso_objective(X, y, θ, λ)
    return 0.5*mean(abs2, X*θ - y) + λ*sum(abs, θ)
end

function proj_max_ball(θ)
    for i in eachindex(θ)
        θ[i] >  1 ? θ[i] =  1.0 :
        θ[i] < -1 ? θ[i] = -1.0 :
        nothing;
    end
    return θ
end

function prox_ℓ₁_norm(θ, κ)
    for i in eachindex(θ)
         abs(θ[i]) > κ ? θ[i] = sign(θ[i])*max(abs(θ[i]) - κ, 0) :
         θ[i] = 0.0
     end
end

function solve_lasso(X, y, θ, λ, θᵀ; itm=1000, tol=1e-6, ptf=100)
    θ⁻ = copy(θ)
    n = size(X, 1)
    p = length(θ)
    ϵ = 1e-8
    η = 0.001
    κ = η*λ

    noi = 0

    true_obj = lasso_objective(X, y, θᵀ, λ)
    obj = lasso_objective(X, y, θ, λ)

    err = obj - true_obj
    while err ≥ ϵ
        Δθ = X'*(X*θ - y)/n
        BLAS.axpy!(-η, Δθ, θ)
        
        prox_ℓ₁_norm(θ, κ)

        obj = lasso_objective(X, y, θ, λ)
        err = obj - true_obj

        noi += 1
        if noi % ptf == 0
            println("err: ", err, "\nobj: ", obj, "\n")
        end
    end
    return θ
end

function solve_fused_lasso(X, y, θ, λ, γ, C, θᵀ; itm=1000, tol=1e-6, ptf=100)
    # fused lasso with smoothing proximal gradient descent
    θ⁻ = copy(θ)
    n = size(X, 1)
    p = length(θ)
    ϵ = 1e-8
    D = (p-1)/2
    μ = ϵ/(2*D)
    L = opnorm(X)^2/n + opnorm(C)^2 / μ
    #η = 1/L
    η = 0.001
    κ = η*λ

    noi = 0
    
    true_obj = objective(X, y, θᵀ, λ, γ)
    obj = objective(X, y, θ, λ, γ)

    err = obj - true_obj

    t = 1
    ϕ = copy(θ)
    ϕ⁻ = copy(θ)
    while err ≥ ϵ
        # gradient step on θ
        α = proj_max_ball(C*θ ./ μ)
        Δθ = X'*(X*θ - y)/n + C'*α
        BLAS.axpy!(-η, Δθ, θ)

        # proximal operation
        prox_ℓ₁_norm(θ, κ)
    
        # learning information
        obj = objective(X, y, θ, λ, γ)
        err = obj - true_obj

        noi += 1
        if noi % ptf == 0 
            println("err: ", err, "\nobj: ", obj, "\n")
        end
    end
    return θ
end
