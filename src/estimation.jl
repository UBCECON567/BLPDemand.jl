
"""
    function share(δ::AbstractVector, σ::AbstractVector, x::AbstractMatrix, ν::AbstractMatrix)

Compute market shares in BLP random coefficients demand model.

Market shares are given by 

```math
 s_{j} = \\int \\frac{e^{\\delta_j + x_j σ ν}}{1+\\sum_{i=1}^J e^{\\delta_i + x_i σ ν}} dF_ν(ν)
```

# Arguments:
- `δ` vector of length `J=` number of products
- `σ` vector of length `K=` number of characteristics
- `x` `K × J` matrix of product characteristics
- `ν` `K × S` matrix of simulation draws for integration

Returns vector of length `J` market shares. 

See also: [`delta`](@ref)
"""
function share(δ::AbstractVector,
               σ::AbstractVector,
               x::AbstractMatrix,
               ν::AbstractMatrix)
  J = length(δ)
  K = length(σ)
  S = size(ν,2)

  s = zeros(eltype(δ),size(δ))
  si = similar(s)
  σx = σ.*x
  @inbounds for i in 1:S
    @simd for j in 1:J
      @views si[j] = exp(δ[j] + dot(σx[:,j], ν[:,i]))
    end
    si ./= (1 + sum(si))
    s .+= si
  end
  s ./= S
  return(s)
end

"""
   function sharep(β::AbstractVector,
                   σ::AbstractVector,
                   p::AbstractVector,
                   x::AbstractMatrix,
                   ν::AbstractMatrix,
                   ξ::AbstractVecvtor)

Compute market shares in BLP random coefficients demand model.

Market shares are given by 

```math
 s_{j} = \\int \\frac{e^{\\delta_j + x_j σ ν}}{1+\\sum_{i=1}^J e^{\\delta_i + x_i σ ν}} dF_ν(ν)
```
where 
```math
\\delta_j = β[1]*p[j] + x[:,j]' *β[2:end] + ξ[j]
```

# Arguments:
- `β` vector of length `K=` number of characteristics
- `σ` vector of length `K=` number of characteristics
- `p` `J` vector of prices
- `x` `(K-1) × J` matrix of exogenous product characteristics
- `ν` `K × S` matrix of simulation draws for integration
- `ξ` `J` vector of demand shocks

Returns vector of length `J` market shares. 

See also: [`share`, `dsharedp`](@ref)  
"""
function sharep(β::AbstractVector,
                σ::AbstractVector,
                p::AbstractVector,
                x::AbstractMatrix,
                ν::AbstractMatrix,
                ξ::AbstractVecvtor)
  @views δ = β[1]*p + x'*β[2:end] + ξ
  J = length(δ)
  K = length(σ)
  S = size(ν,2)

  s = zeros(eltype(δ),size(δ))
  si = similar(s)
  @views σx = σ[2:end].*x
  @inbounds for i in 1:S
    @simd for j in 1:J
      @views si[j] = exp(δ[j] + σ[1]*p[j]*ν[1,i] + dot(σx[:,j], ν[2:end,i]))
    end
    si ./= (1 + sum(si))
    s .+= si
  end
  s ./= S
  return(s)
end


"""
    function dsharedp(β::AbstractVector,
                  σ::AbstractVector,
                  p::AbstractVector,
                  x::AbstractMatrix,
                  ν::AbstractMatrix,
                  ξ::AbstractVector)

Compute market shares and their derivatives in BLP random coefficients demand model.

See [`sharep`](@ref) for argument details.

# Returns 
- `s` vector `J` market shares
- `ds` `J × J` Jacobian matrix with ds[l,j] = ∂s[l]/∂p[j]
- `Λ` `J × J` diagonal matrix with `ds = Λ - Γ`
- `Γ` `J × J` matrix 

See [`eqprices`](@ref) for usage of `Λ` and `Γ`.    
"""
function dsharedp(β::AbstractVector,
                  σ::AbstractVector,
                  p::AbstractVector,
                  x::AbstractMatrix,
                  ν::AbstractMatrix,
                  ξ::AbstractVector)
  @views δ = β[1]*p + x'*β[2:end] + ξ
  J = length(δ)
  K = length(σ)
  S = size(ν,2)

  s = zeros(eltype(δ),size(δ))
  Γ = zeros(eltype(δ), length(δ), length(p))
  Λ = zeros(eltype(δ), length(δ))
  si = similar(s)
  @views σx = σ[2:end].*x
  @inbounds for i in 1:S
    @simd for j in 1:J
      @views si[j] = exp(δ[j] + σ[1]*p[j]*ν[1,i] + dot(σx[:,j], ν[2:end,i]))
    end
    si ./= (1 + sum(si))
    s .+= si
    for j in 1:J
      Λ[j] += si[j]*(β[1] + σ[1]*ν[1,i])
      for l in 1:J
        Γ[l,j] += si[l]*si[j]*(β[1] + σ[1]*ν[1,i])
      end
    end
  end
  s ./= S
  Λ ./= S
  Γ ./= S
  ds = Diagonal(Λ) .- Γ
  return(s=s, ds=ds, Λ=Diagonal(Λ), Γ=Γ)
end


"""

function delta(s, x, ν, σ; 
               tol=sqrt(eps(eltype(s))), maxiter=1000)
  
    
Solves for δ in s = share(δ, ...) using contraction mapping iteration.
 
See also: [`share`](@ref)
"""
function delta(s::AbstractVector,
               x::AbstractMatrix,
               ν::AbstractMatrix,
               σ::AbstractVector;
               tol=sqrt(eps(eltype(s))), maxiter=1000)

  δ = log.(s) .- log.(1-sum(s))
  δold = copy(δ)
  smodel = copy(s)  
  normchangeδ = 10*tol
  normserror = 10*tol
  iter = 0
  while (normchangeδ > tol) && (normserror > tol) && iter < maxiter
    δ, δold = δold, δ
    smodel .= share(δold, σ, x, ν)
    smodel .= max.(smodel, eps(0.0)) # avoid log(negative)
    δ .= δold .+ log.(s) .- log.(smodel)
    normchangeδ = norm(δ - δold)
    normserror = norm(s - smodel)
    iter += 1
  end

  if (iter>maxiter)
    @warn "Maximum iterations ($maxiter) reached"
  end
  
  return((δ = δ, normchangeδ=normchangeδ, normserror=normserror, iter=iter))  
end

"""
    function demandmoments(β::AbstractVector,σ::AbstractVector,
                           s::AbstractMatrix, 
                           x, ν, ivdemand)

Demand side moments in BLP model. 

# Arguments

- `β` length `K` (=number characteristics) vector of average tastes for characteristics
- `σ` length `K` vector of standard deviations of tastes
- `s` `J × T` matrix of market shares
- `x` `K × J × T` array of product characteristics
- `ν` `K × S × T` array of draws of `ν`
- `ivdemand` `L × J × T` array of instruments. Identification requries `L ≥ K`.

Returns `L × J` array of moments.
  
See also: [`share`](@ref), [`delta`](@ref), [`simulateRCIVlogit`](@ref)
"""
function demandmoments(β::AbstractVector,σ::AbstractVector,
                       s::AbstractMatrix, 
                       x, ν, ivdemand)
  # compute δ 
  ξ = similar(s)
  for t in 1:size(s,2)
    @views ξ[:,t] = delta(s[:,t], x[:,:,t], ν[:,:,t], σ).δ .- x[:,:,t]' * β
  end
  
  moments = similar(ξ, size(ξ, 1), size(ivdemand,1))
  for j in 1:size(ξ, 1)
    for k in 1:size(ivdemand,1)
      @views moments[j,k] = sum(ξ[j,:].*ivdemand[k,j,:])/size(ξ,2)
    end
  end
  
  #obj = size(ξ,2)*moments[:]'*W*moments[:]r
  return(moments)
end


"""
    function supplymoments(γ::AbstractVector, β::AbstractVector, σ::AbstractVector,
                           ξ::AbstractVector, x, ν, w, ivsupply; 
                           firmid= (1:size(ξ,1)) .* fill(1, size(ξ,2))')

Supply side moments in a BLP model. Assumes that marginal cost is log linear, 
```math
c_{jt} = \\exp(w_{jt}'γ + ω_{jt})
```
and prices are Bertrand-Nash
```math
c_{t} = p_{t} + (∂s/∂p)^{-1} s
```
where `∂s/∂p` is the Jacobian of shares with respect to prices, but with the `j,l` entry set to zero unless goods `j` and `l` are produced by the same firm. 

# Arguments

- `γ` marginal cost coefficients
- `β` length `K` (=number characteristics) vector of average tastes for characteristics
- `σ` length `K` vector of standard deviations of tastes
- `ξ` `J × T` matrix of market demand shocks
- `x` `K × J × T` array of product characteristics
- `ν` `K × S × T` array of draws of `ν`
- `w` `length(γ) × J × T` array of cost shifters
- `ivsupply` `M × J × T` array of instruments. Identification requries `M ≥ length(γ)`.
- `firmid= (1:J) .* fill(1, T)'` identifier of firm producing each good. Default value assumes each good is produced by a different firm. 

Returns `L × J` array of moments.
"""
function supplymoments(γ::AbstractVector, β::AbstractVector, σ::AbstractVector,
                       ξ::AbstractMatrix, x, ν, w, ivsupply;
                       firmid= (1:size(ξ,1)) .* fill(1, size(ξ,2))',
                       endogenousx=1)

  J, T = size(ξ)
  K = length(β)
  # pre-allocate arrays
  ω = similar(γ, size(ξ))
  p = similar(x, J)
  
  for t in 1:T
    p .= x[1,:,t]
    @views s, Js, Λ, Γ = dsharedp(β, σ, p, x[2:end,:,t], ν[:,:,t], ξ[:,t])
    Js .= Js .* (firmid[:,t].==firmid[:,t]')
    @views p .= p .+ Js \ s
    @views ω[:, t] .= log.(p) .- w[:,:,t]'*γ
  end

  moments = similar(ω, J, size(ivsupply,1))
  for j in 1:J
    for k in 1:size(ivsupply,1)
      @views moments[j,k] = sum(ω[j,:].*ivsupply[k,j,:])/size(ω,2)
    end
  end
  
  return(moments)  
end

"""
    function eqprices(mc::AbstractVector,
                      β::AbstractVector, σ::AbstractVector,
                      ξ::AbstractVector,
                      x, ν;
                      firmid= 1:length(mc),
                      tol=sqrt(eps(eltype(mc))),
                      maxiter=10000)

Compute equilibrium prices in BLP model using ζ contraction method of [Morrow & Skerlos (2011)](
https://www.jstor.org/stable/23013173). 

# Arguments

- `mc` vector of `J` marginal costs
- `β` vector of `K` taste coefficients
- `σ` vector of `K` taste standard deviations
- `ξ` vector of `J` demand shocks
- `x` `(K-1) × J` exogenous product characteristics
- `ν` `K × S × T` array of draws of `ν`
- `firmid= (1:J) .* fill(1, T)'` identifier of firm producing each good. Default value assumes each good is produced by a different firm. 
- `tol` convergence tolerance
- `maxiter` maximum number of iterations.

"""
function eqprices(mc::AbstractVector,
                  β::AbstractVector, σ::AbstractVector,
                  ξ::AbstractVector,
                  x, ν;
                  firmid= 1:length(mc),
                  tol=sqrt(eps(eltype(mc))),
                  maxiter=10000, verbose=0)

  iter = 0
  dp = 10*tol
  focnorm = 10*tol
  p = mc*1.1
  pold = copy(p)
  samefirm = firmid.==firmid'  
  while (iter < maxiter) && ((dp > tol) || (focnorm > tol))
    s, ds, Λ, Γ = dsharedp(β, σ, p, x, ν, ξ)    
    ζ = inv(Λ)*(samefirm.*Γ)*(p - mc) - inv(Λ)*s
    focnorm = norm(Λ*(p-mc - ζ))
    pold, p = p, pold
    p .= mc .+ ζ
    dp = norm(p-pold)
    if verbose && (iter % 100 == 0)
      @show iter, p, focnorm
    end
    iter += 1    
  end
  if verbose
    @show iter, p, focnorm
  end
  return(p)  
end
