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
  ξ = zeros(promote_type(eltype(x), eltype(β)), size(s))
  for t in 1:size(s,2)
    @views ξ[:,t] = delta(s[:,t], x[:,:,t], ν[:,:,t], σ) .- x[:,:,t]' * β
  end
  
  moments = similar(ξ, size(ξ, 1), size(ivdemand,1))
  for j in 1:size(ξ, 1)
    for k in 1:size(ivdemand,1)
      @views moments[j,k] = sum(ξ[j,:].*ivdemand[k,j,:])/size(ξ,2)
    end
  end
  
  #obj = size(ξ,2)*moments[:]'*W*moments[:]
  return((moments=moments, ξ=ξ))
end

function demandmoments(β::AbstractVector,
                       σ::AbstractVector,
                       ξ::AbstractMatrix, 
                       s::AbstractMatrix, 
                       x, ν, ivdemand)
  # compute δ 
  #ξ = similar(s)
  #for t in 1:size(s,2)
  #  @views ξ[:,t] = delta(s[:,t], x[:,:,t], ν[:,:,t], σ) .- x[:,:,t]' * β
  #end
  
  moments = similar(ξ, size(ξ, 1), size(ivdemand,1))
  for j in 1:size(ξ, 1)
    for k in 1:size(ivdemand,1)
      @views moments[j,k] = sum(ξ[j,:].*ivdemand[k,j,:])/size(ξ,2)
    end
  end
  
  #obj = size(ξ,2)*moments[:]'*W*moments[:]
  return((moments=moments, ξ=ξ))
end


function safelog(x)
  δ = 1e-8
  if (x<δ)
    log(δ) + (δ-x)/δ
  else
    log(x)
  end
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
  #@show β[1], minimum(ν[1,:,:]*σ[1]), maximum(ν[1,:,:]*σ[1])
  for t in 1:T
    p .= x[1,:,t]
    @views s, Js, Λ, Γ = dsharedp(β, σ, p, x[2:end,:,t], ν[:,:,t], ξ[:,t])
    Js .= Js .* (firmid[:,t].==firmid[:,t]')
    mc = p + Js \ s
    @views ω[:, t] .= safelog.(mc) .- w[:,:,t]'*γ
  end
  moments = similar(ω, J, size(ivsupply,1))
  #mi = similar(ω, length(moments), size(ω, 2))
  for j in 1:J
    for k in 1:size(ivsupply,1)
      @views moments[j,k] = sum(ω[j,:].*ivsupply[k,j,:])/size(ω,2)
      #@views mi[LinearIndices(moments)[j,k], :] .= ω[j,:].*ivsupply[k,j,:]      
    end
  end  
  return((moments=moments, ω=ω))
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
    function makeivblp(x::AbstractMatrix, firmid=1:size(x,2))

Function for constructing demand instrumental variables following BLP (1995).

Instruments consist of exogenous product characteristics, sum of rivals exogenous characteristics, and, if there are any multi-product firms, sum of characteristics of other goods produced by the same firm.
"""
function makeivblp(x::AbstractMatrix, firmid=1:size(x,2))
  K,J = size(x)
  if length(firmid)==length(unique(firmid))
    ivdemand = similar(x, 2*K, J)
  else
    ivdemand = similar(x, 3*K, J)    
  end
  ivdemand[1:K,:] .= x
  for j in 1:J    
    ivdemand[(K+1):(2*K),j] .= vec(sum(x[:,firmid[j] .!= firmid], dims=2))
  end
  if size(ivdemand,2)>=3*K
    for j in 1:J
      ivdemand[(2*K+1):(3*K), j] .= vec(sum(x[:, firmid[j] .== firmid], dims=2)) .- x[:,j]
    end
  end
  
  return(ivdemand)  
end

function makeivblp(x::Array{R, 3}, firmid=(1:size(x,2)) .* fill(1, size(x,3))') where R
  K,J,T = size(x)
  if size(firmid,1)==length(unique(firmid))
    ivdemand = similar(x, 2*K, J, T)
  else
    ivdemand = similar(x, 3*K, J, T)    
  end
  for t in 1:T
    ivdemand[:,:,t] .= makeivblp(x[:,:,t], firmid[:,t])
  end
  return(ivdemand)  
end


function pack(β::AbstractVector, σ::AbstractVector, γ::AbstractVector)
  θ = vcat(β, log.(σ), γ)
  unvec = let K=length(β)
    θ->(β=θ[1:K], σ=exp.(θ[(K+1):(2*K)]), γ=θ[(2*K+1):end])
  end
  (θ=θ, unpack=unvec)
end

"""
   pack(β::AbstractVector, σ::AbstractVector, ...)

Packs parameters into a single vector, θ.

Returns tuple with the packed parameters, θ, and a function to unpack them, i.e.
```
unpack(θ) = (β, σ, ...)
```
and
```
pack(unpack(θ)) = θ
```
"""
function pack(β::AbstractVector, σ::AbstractVector)
  θ = vcat(β, σ)
  K = length(β)
  unvec = let K=length(β)
    θ->(β=θ[1:K], σ=θ[(K+1):(2*K)])
  end
  (θ=θ, unpack=unvec)
end

function pack(β::AbstractVector, σ::AbstractVector, ξ::AbstractMatrix)
  θ = vcat(β, σ, ξ[:])
  J, T = size(ξ)
  K = length(β)
  unvec = let K=length(β), J=J, T=T
    θ->(β=θ[1:K], σ=θ[(K+1):(2*K)], ξ=reshape(θ[(2K+1):end], J,T) )
  end
  (θ=θ, unpack=unvec)
end

"""
    objectiveRCIVlogit(β, σ, s, x, ν, ivdemand; W=I)

GMM objective function for random coefficients IV logit model. 
"""
function objectiveRCIVlogit(β, σ, s, x, ν, ivdemand; W=I)
  (m, ξ) = demandmoments(β, σ, s, x, ν, ivdemand)
  return(size(s,2)*m[:]'*W*m[:])
end


""" 
    estimateRCIVlogit(s, x, ν, iv, method=:MPEC)

Estimates a random coefficients IV logit model. 

# Arguments
- `s` `J × T` matrix of market shares
- `x` `K × J × T` array of product characteristics
- `ν` `K × S × T` array of draws for MC integration
- `iv` `L × J × T` array of instruments
- `method` method for estimation. Available choices are :MPEC or :NFXP.
- `verbose` whether to display information about optimization progress
"""
function estimateRCIVlogit(s, x, ν, iv; method=:MPEC, verbose=true)
    
  K, J, T = size(x)
  L = size(iv, 1)
  σ0 = ones(K)

  # initial β from logit
  Y = reshape(log.(s) .- log.(1 .- sum(s,dims=1)), J*T)
  X = reshape(x, K,J*T)
  Z = reshape(iv, L,J*T)
  xz=((Z*Z') \ Z*X')'*Z
  β0 = (xz*xz') \ xz*Y

  if method==:NFXP
    θ0, unpack = pack(β0, σ0)
    obj(θ) = objectiveRCIVlogit(unpack(θ)..., s, x, ν, iv)
    opt = optimize(obj, θ0, method=LBFGS(),show_trace=verbose, autodiff=:forward)
    β, σ = unpack(opt.minimizer)
    m, ξ = demandmoments(β,σ, s, x, ν, iv)
    out = (β=β, σ=σ, ξ=ξ, opt=opt)
  elseif method==:MPEC
    mod = Model()
    K,J,T = size(x)
    S = size(ν,2)
    @variable(mod, β[1:K])
    @variable(mod, σ[1:K] ≥ 0)
    @variable(mod, ξ[1:J,1:T])
    njit = @NLexpression(mod, [j in 1:J, i in 1:S, t in 1:T], #exp(δi[j,i,t]))
                         exp(sum(x[k,j,t]*β[k] for k in 1:K) + ξ[j,t] + sum(σ[k]*ν[k, i, t]*x[k,j,t] for k in 1:K)))
    dit = @NLexpression(mod, [i in 1:S, t in 1:T], 1 + sum(njit[j,i,t] for j in 1:J))
    sjit = @NLexpression(mod, [j in 1:J, i in 1:S, t in 1:T], njit[j,i,t]/dit[i,t])
    @NLconstraint(mod, share[j in 1:J, t in 1:T], s[j,t] == sum(sjit[j,i,t] for i in 1:S)/S)
    @objective(mod, Min, sum( sum( ξ[j, t] * iv[l, j ,t] for t in 1:T)^2/T
                              for l in 1:size(iv,1), j in 1:J ));
    set_start_value.(mod[:β], β0)
    set_start_value.(mod[:σ], σ0)
    # start from a feasible point
    ξ0 = similar(s)
    for t in 1:T
      ξ0[:,t] = delta(s[:,t], x[:,:,t], ν[:,:,t], start_value.(mod[:σ])) - x[:,:,t]'*start_value.(mod[:β])
    end
    set_start_value.(mod[:ξ], ξ0)
    
    set_optimizer(mod,  with_optimizer(Ipopt.Optimizer,
                                       print_level=5*verbose,
                                       max_iter=1000))
    optimize!(mod)
    out = (β=value.(mod[:β]), σ=value.(mod[:σ]), ξ=value.(mod[:ξ]), opt=mod)
  else
    error("method $method not recognized")
  end
  return(out)    
end




""" 
    estimateRCIVlogit(s, x, ν, iv, method=:MPEC)

Estimates a random coefficients IV logit model. 

# Arguments
- `s` `J × T` matrix of market shares
- `x` `K × J × T` array of product characteristics. First one must be price
- `ν` `K × S × T` array of draws for MC integration
- `ivdemand` `L × J × T` array of instruments
- `w` array of cost shifters
- `ivsupply` array of instruments
- `method` method for estimation. Available choices are :MPEC or :NFXP.
- `verbose` whether to display information about optimization progress
"""
function estimateBLP(s::AbstractMatrix, #p::AbstractMatrix,
                     x::AbstractArray{T, 3} where T, ν::AbstractArray{T,3} where T,
                     ivdemand::AbstractArray{T,3} where T,
                     w::AbstractArray{T, 3} where T, ivsupply::AbstractArray{T, 3} where T;
                     method=:MPEC, verbose=true, firmid=1:size(s,2))

  smalls = 1e-4
  if (minimum(s) < smalls)
    @warn "There are $(sum(s .< smalls)) shares < $smalls."
    @warn "Estimation may encounter numeric problems with small shares."
  end
  if (maximum(s) > 1.0 - smalls)
    @warn "There are $(sum(s .> 1.0 - smalls)) shares > 1 - $smalls."
    @warn "Estimation may encounter numeric problems with shares near 1."
  end

  
  K, J, T = size(x)
  L = size(ivdemand, 1)
  σ0 = ones(K)

  # initial β from logit
  Y = reshape(log.(s) .- log.(1 .- sum(s,dims=1)), J*T)
  X = reshape(x, K,J*T)
  Z = reshape(ivdemand, L,J*T)
  xz=((Z*Z') \ Z*X')'*Z
  β0 = (xz*xz') \ xz*Y

  # initial γ
  γ0 = zeros(size(w, 1))
  m, ξ = demandmoments(β0, 0*σ0, s, x, ν, ivdemand)
  m, ω = supplymoments(γ0, β0, 0*σ0, ξ, x, ν, w, ivsupply)
  Y = reshape(ω, J*T)
  X = reshape(w, size(w,1), J*T)
  γ0 = X' \ Y  
  @show β0, σ0, γ0
  
  if method==:NFXP
    θ0, unpack = pack(β0, σ0, γ0)
    objectiveBLP = let s=s, px=x, ν=ν, zd=ivdemand, w=w, zs=ivsupply, T=T
      function(θ)
        β, σ, γ = unpack(θ)
        md, ξ = demandmoments(β,σ, s, px, ν, zd)
        ms, ω = supplymoments(γ, β, σ, ξ, px, ν, w, zs)
        return(sum(T*(dot(md,md) + dot(ms,ms))))
      end
    end
    @show objectiveBLP(θ0)
    opt = optimize(objectiveBLP, θ0, method=LBFGS(),show_trace=verbose, autodiff=:forward)
    β, σ, γ = unpack(opt.minimizer)
    m, ξ = demandmoments(β,σ, s, x, ν, ivdemand)
    m, ω = supplymoments(γ, β, σ, ξ, x, ν, w, ivsupply)
    out = (β=β, σ=σ, γ=γ, ξ=ξ,ω=ω, opt=opt)
  elseif method==:MPEC
    mod = Model()
    K,J,T = size(x)
    Kw = size(w,1)
    S = size(ν,2)
    @variable(mod, β[1:K])
    @variable(mod, σ[1:K] ≥ 0)
    @variable(mod, γ[1:Kw])
    @variable(mod, ξ[1:J,1:T])
    @variable(mod, ω[1:J,1:T])
    njit = @NLexpression(mod, [j in 1:J, i in 1:S, t in 1:T], #exp(δi[j,i,t]))
                         exp(sum(x[k,j,t]*β[k] for k in 1:K) + ξ[j,t] + sum(σ[k]*ν[k, i, t]*x[k,j,t] for k in 1:K)))
    dit = @NLexpression(mod, [i in 1:S, t in 1:T], 1 + sum(njit[j,i,t] for j in 1:J))
    sjit = @NLexpression(mod, [j in 1:J, i in 1:S, t in 1:T], njit[j,i,t]/dit[i,t])
    @NLconstraint(mod, share[j in 1:J, t in 1:T], s[j,t] == sum(sjit[j,i,t] for i in 1:S)/S)
    
    Λ = @NLexpression(mod, [j in 1:J, t in 1:T], sum(sjit[j,i,t]*(β[1]+σ[1]*ν[1,i,t]) for i in 1:S)/S)
    Γ = @NLexpression(mod, [j in 1:J, jj in 1:J, t in 1:T],
                      (firmid[j]==firmid[jj])* sum(sjit[j,i,t]*sjit[jj,i,t]*(β[1]+σ[1]*ν[1,i,t]) for i in 1:S)/S)
    mc = @NLexpression(mod, [j in 1:J, t in 1:T], exp(ω[j,t] + sum(w[l,j,t]*γ[l] for l in 1:Kw)))
    @NLconstraint(mod, foc[j in 1:J, t in 1:T], 0 == s[j,t]/Λ[j,t] + 
                  x[1,j,t]-mc[j,t] - sum( (x[1,jj,t] - mc[jj,t])*Γ[j,jj,t]/Λ[jj,t]
                                          for jj in findall(firmid[j].==firmid)) )
    @objective(mod, Min, sum( sum( ξ[j, t] * ivdemand[l, j ,t] for t in 1:T)^2/T
                              for l in 1:size(ivdemand,1), j in 1:J )
               + sum( sum( ω[j, t] * ivsupply[l, j ,t] for t in 1:T)^2/T
                      for l in 1:size(ivsupply,1), j in 1:J )
               );
    set_start_value.(mod[:β], β0)
    set_start_value.(mod[:σ], σ0)
    set_start_value.(mod[:γ], 0)
    # start from a feasible point
    ξ0 = similar(s)
    for t in 1:T
      ξ0[:,t] = delta(s[:,t], x[:,:,t], ν[:,:,t], start_value.(mod[:σ])) - x[:,:,t]'*start_value.(mod[:β])
    end
    set_start_value.(mod[:ξ], ξ0)
    set_start_value.(mod[:ω], 0)
    
    set_optimizer(mod,  with_optimizer(Ipopt.Optimizer,
                                       print_level=5*verbose,
                                       max_iter=1000))
    optimize!(mod)
    out = (β=value.(mod[:β]), σ=value.(mod[:σ]), γ=value.(mod[:γ]),
           ξ=value.(mod[:ξ]), ω=value.(mod[:ω]), opt=mod)
  else
    error("method $method not recognized")
  end
  return(out)    
end

