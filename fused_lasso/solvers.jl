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

function group_lasso_objective(X, y, θ, λ)
    s1 = 15 * norm(θ[1:15])
    s2 = 5 * norm(θ[16:20])
    s3 = 30 * norm(θ[21:50])
    s4 = 10 * norm(θ[51:60])
    s5 = 29 * norm(θ[61:89])
    s6 = 7 * norm(θ[90:96])
    s7 = 4 * norm(θ[97:100])
    return 0.5*mean(abs2, X*θ - y) + λ*(s1 + s2 + s3 + s4 + s5 + s6 + s7)
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

function prox_ℓ₁₂_norm(θ, κ)
    ρ = norm(θ)
    r = ρ
    ρ == 0 && return;
    ρ > κ ? r -= κ : r = 0.0
    θ = (r/ρ) * θ
    return θ
end

function solve_lasso(X, y, θ, λ, θᵀ; itm=1000, tol=1e-6, ptf=100)
    θ⁻ = copy(θ)
    n = size(X, 1)
    p = length(θ)
    ϵ = 1e-10
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

function solve_group_lasso(X, y, θ, λ, θᵀ; itm=1000, tol=1e-6, ptf=100)
    θ⁻ = copy(θ)
    n = size(X, 1)
    p = length(θ)
    ϵ = 1e-10 
    η = 0.001
    κ = η*λ

    noi = 0
    
    true_obj = group_lasso_objective(X, y, θᵀ, λ)
    obj = group_lasso_objective(X, y, θ, λ)

    err = obj - true_obj
    while err ≥ ϵ
        Δθ = X'*(X*θ - y)/n
        BLAS.axpy!(-η, Δθ, θ)
    
        θ[1:15] = prox_ℓ₁₂_norm(θ[1:15], κ*15)
        θ[16:20] = prox_ℓ₁₂_norm(θ[16:20], κ*5)
        θ[21:50] = prox_ℓ₁₂_norm(θ[21:50], κ*30)
        θ[51:60] = prox_ℓ₁₂_norm(θ[51:60], κ*10)
        θ[61:89] = prox_ℓ₁₂_norm(θ[61:89], κ*29)
        θ[90:96] = prox_ℓ₁₂_norm(θ[90:96], κ*7)
        θ[97:100] = prox_ℓ₁₂_norm(θ[97:100], κ*4)

        obj = group_lasso_objective(X, y, θ, λ)
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
    ϵ = 1e-10
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
