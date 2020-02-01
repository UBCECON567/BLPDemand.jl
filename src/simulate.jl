"""
    function simulateIVRClogit(T, β, σ, π, ρ, S)  

Simulates a random coefficients logit model with endogeneity.

# Arguments

- `T::Integer` number of markets
- `β::AbstractVector` with `length(β)=K`, average tastes for characteristics.
- `σ::AbstractVector` with `length(σ)=K`, standard deviation of tastes for characteristics
- `π::AbstractMatrix` with `size(π) = (L, K, J)`, first stage coefficients
- `ρ::Number` strength of endogeneity
- `S` number of simulation draws to calculate market shares

# Returns

- `x` array of product characteristics with `size(x) = (K, J, T)`
- `z` array of instruments with `size(z) = (L, J, T)`
- `s` matrix of market shares `size(s) = (J,T)`
- `ν` random draws used to compute market shares, `size(ν) = (K,S,T)`
- `ξ` market demand shocks, `size(ξ) = (J,T)`
"""
function simulateIVRClogit(T, β, σ, π, ρ, S; varξ=1)  
  (niv, nchar, J) = size(π)
  x = zeros(nchar, J, T)
  ξ = zeros(J,T)
  z = randn(niv, J, T)
  endo = randn(length(β), T)
  for j in 1:J
    x[:,j,:] = π[:,:,j]'*z[:,j,:] .+ endo
    ξ[j,:] = (randn(T)*sqrt(1-ρ^2) .+ endo[1,:].*ρ)*varξ
  end
  ν = randn(nchar, S, T)
  s = zeros(J,T)
  for t in 1:T
    @views s[:,t] .= share(x[:,:,t]'*β+ξ[:,t], σ, x[:,:,t], ν[:,:,t])
  end
  return((x=x, z=z, s=s, ν=ν, ξ=ξ))
end

"""
    function simulateBLP(T, β, σ, γ, S)  

Simulates a BLP demand and supply model.

# Arguments

- `T::Integer` number of markets
- `β::AbstractVector` with `length(β)=K`, average tastes for characteristics. The first characteristic will be endogeneous (price)
- `σ::AbstractVector` with `length(σ)=K`, standard deviation of tastes for characteristics
- `γ::AbstractVector` marginal cost coefficients
- `S` number of simulation draws to calculate market shares

# Returns

- `x` array of product characteristics with `size(x) = (K, J, T)`
- `z` array of instruments with `size(z) = (L, J, T)`
- `s` matrix of market shares `size(s) = (J,T)`
- `ν` random draws used to compute market shares, `size(ν) = (K,S,T)`. `log(-ν[1,:,:])` is N(0,1) the other components of ν are N(0,1). It is important that β[1] + σ[1]*ν[1,:,:] is negative.
- `ξ` market demand shocks, `size(ξ) = (J,T)`
"""
function simulateBLP(J, T, β::AbstractVector, σ::AbstractVector, γ::AbstractVector, S;
                     varξ=1, varω=1)
  
  K = length(β)
  x = rand(K, J, T)
  ξ = randn(J,T)*varξ
  L = length(γ)
  w = rand(L, J, T)
  ν = randn(K, S, T)
  ν[1,:,:] .= -rand(S,T) # make sure individuals price coefficients are negative
  ω = randn(J,T)*varω
  s = zeros(J,T)
  p = zeros(J,T)

  for t in 1:T
    c = exp.(w[:,:,t]'*γ + ω[:,t])
    p[:,t] .= eqprices(c, β, σ, ξ[:,t], x[2:end,:,t], ν[:,:,t])

    x[1,:,t] .= p[:,t]
    s[:,t] .= share(x[:,:,t]'*β+ξ[:,t], σ, x[:,:,t], ν[:,:,t])
  end

  
  return((x=x, w=w, p=p, s=s, ν=ν, ξ=ξ, ω=ω))
  
end

