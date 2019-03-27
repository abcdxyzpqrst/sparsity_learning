using LinearAlgebra

# TODO: FLSA, Smooth-Prox Comparison
function objective(X, y, θ, λ, μ)
    s = 0.0                         # fusion penalty 
    for j = 2:length(θ)
        s += abs(θ[j] - θ[j-1])
    end
    return 0.5*sum(abs2, X*θ - y) + λ*sum(abs, θ) + μ*s
end


