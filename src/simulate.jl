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

Returns [`BLPData`](@ref) struct
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
  return(BLPData(s, x, ν, z))
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
- `firmid= (1:J)` identifier of firm producing each good. Default value assumes each good is produced by a different firm. 
- `tol` convergence tolerance
- `maxiter` maximum number of iterations.

"""
function eqprices(mc::AbstractVector,
                  β::AbstractVector, σ::AbstractVector,
                  ξ::AbstractVector,
                  x::AbstractMatrix, ν::AbstractMatrix;
                  firmid= 1:length(mc),
                  tol=sqrt(eps(eltype(mc))),
                  maxiter=10000, verbose=false)

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

"""
    function simulateBLP(J, T, β, σ, γ, S; varξ=1, varω=1, randj=1.0, firmid=1:J, costf=:log)

Simulates a BLP demand and supply model.

# Arguments

- `J` numebr of products
- `T::Integer` number of markets
- `β::AbstractVector` with `length(β)=K`, average tastes for characteristics. The first characteristic will be endogeneous (price)
- `σ::AbstractVector` with `length(σ)=K`, standard deviation of tastes for characteristics
- `γ::AbstractVector` marginal cost coefficients
- `S` number of simulation draws to calculate market shares
- `varξ=1` standard deviation of ξ
- `varω=1` standard deviation of ω
- `randj=1.0` if less than 1, then J is the maximum number of products per market. Each product is included in each market with probability randj
- `firmid=1:J` firm identifiers. Must be length J. 

# Returns 
- `dat` a [`BLPData`](@ref) struct
- `ξ`
- `ω`
"""
function simulateBLP(J, T, β::AbstractVector, σ::AbstractVector, γ::AbstractVector, S;
                     varξ=1, varω=1, randj=1.0, firmid=1:J, costf=:log)
  
  K = length(β)
  L = length(γ)
  dat = Array{MarketData,1}(undef, T)
  Ξ = Array{Vector,1}(undef,T)
  Ω = Array{Vector,1}(undef,T)
  for t in 1:T
    Jt, fid = if randj < 1.0
      inc = falses(J)
      while sum(inc)==0
        inc .= rand(J) .< randj
      end
      sum(inc), firmid[inc]
    else
      J, firmid
    end
    x = rand(K, Jt)
    ξ = randn(Jt)*varξ
    w = rand(L, Jt)
    ν = randn(K, S)
    ν[1,:] .= -rand(S) # make sure individuals' price coefficients are negative
    ω = randn(Jt)*varω

    c = costf == :log ? exp.(w'*γ + ω) : w'*γ + ω
    p = eqprices(c, β, σ, ξ, x[2:end,:], ν, firmid=fid)

    x[1,:] .= p
    s = share(x'*β+ξ, σ, x, ν)
    z = makeivblp(cat(x[2:end,:],w, dims=1), firmid=fid,
                  forceown=(length(firmid)!=length(unique(firmid))))
    dat[t] = MarketData(s, x, w, fid, z, z, ν)
    Ξ[t] = ξ
    Ω[t] = ω
  end

  return(dat=dat, ξ=Ξ, ω=Ω)
  
end

