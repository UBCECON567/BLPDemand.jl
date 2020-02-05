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

Returns `(moments, ξ)` where vector of moments is length `L` with `moments[l] = 1/(JT) ∑ⱼ∑ₜ ξ[j,t]*ivdemands[l,j,t]`
  
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
  
  moments = similar(ξ, size(ivdemand,1))
  for k in 1:size(ivdemand,1)
    @views moments[k] = sum(ξ.*ivdemand[k,:,:])/length(ξ)
  end
  
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

Returns `(moments, ω)` an `L` vector of moments with `moments[l] = 1/(JT) ∑ⱼ∑ₜ ω[j,t]*ivsupply[l,j,t]` 

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
  moments = similar(ω, size(ivsupply,1))
  #mi = similar(ω, length(moments), size(ω, 2))
  for k in 1:size(ivsupply,1)
    @views moments[k] = sum(ω.*ivsupply[k,:,:])/length(ω)
    #@views mi[LinearIndices(moments)[j,k], :] .= ω[j,:].*ivsupply[k,j,:]      
  end
  return((moments=moments, ω=ω))
end


"""
    function makeivblp(x::AbstractMatrix, firmid=1:size(x,2))

Function for constructing demand instrumental variables following BLP (1995).

Instruments consist of exogenous product characteristics, sum of
rivals exogenous characteristics, and, if there are any multi-product
firms, sum of characteristics of other goods produced by the same
firm.
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
  θ = vcat(β, log.(σ))
  K = length(β)
  unvec = let K=length(β)
    θ->(β=θ[1:K], σ=exp.(θ[(K+1):(2*K)]))
  end
  (θ=θ, unpack=unvec)
end

function pack(β::AbstractVector, σ::AbstractVector, γ::AbstractVector)
  θ = vcat(β, log.(σ), γ)
  unvec = let K=length(β)
    θ->(β=θ[1:K], σ=exp.(θ[(K+1):(2*K)]), γ=θ[(2*K+1):end])
  end
  (θ=θ, unpack=unvec)
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
function estimateRCIVlogit(s, x, ν, iv; method=:MPEC, verbose=true, W=I)
    
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
    obj(θ) = let s=s, x=x, ν=ν, ivdemand=iv, W=W, unpack=unpack
      β, σ = unpack(θ)
      (m, ξ) = demandmoments(β, σ, s, x, ν, ivdemand)
      return(size(s,2)*m[:]'*W*m[:])
    end
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
    ci = CartesianIndices((size(iv,1),J))
    @expression(mod, moment[m in 1:M],
                sum(ξ[ci[m][2],t]*iv[ci[m],t] for t in 1:T)/T)
    @objective(mod, Min, T*moment'*W*moment)
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
    estimateBLP(s, x, ν, ivdemand, w, ivsupply; method=:MPEC, verbose=true, firmid, W)

Estimates a random coefficients BLP demand model

# Arguments
- `s` `J × T` matrix of market shares
- `x` `K × J × T` array of product characteristics. First one must be price.
- `ν` `K × S × T` array of draws for Monte Carlo integration
- `ivdemand` `L × J × T` array of instruments
- `w` `C × J × Tarray of cost shifters
- `ivsupply` `L × J × T` array of instruments. Identification requires `L ≥ 2K + C`
- `method=:MPEC` method for estimation. Available choices are :MPEC, :NFXP, or :GEL.
- `verbose=true` whether to display information about optimization progress
- `firmid` `J` vector of firm identifies. Default value corresponds to J single product firms.
- `W=I` `L × L` weighting matrix for moments. 

# Details

Uses `L` unconditional moments for estimation the moments are
`moments[l] = 1/(JT) ∑ⱼ∑ₜ (ξ[j,t]*ivdemands[l,j,t] + ω[j,t]*ivsupply[l,j,t])`

# Methods
- `:NFXP` nested fixed point GMM. `minimize_θ G(δ(θ),θ)'W G(δ(θ),θ)`
- `:MPEC` constrainted GMM. `minimize_{θ,Δ} G(Δ, θ)' W G(Δ, θ) s.t. Δ = δ(θ)`
- `:GEL` constrained GEL `maximize_{p, θ, Δ} ∑ₜ log(p[t]) s.t. E_p[g(Δ, θ)] = 0 and Δ = δ(θ)`

See also: [`optimalIV`](@ref), [`varianceBLP`](@ref), [`simulateBLP`](@ref)
"""
function estimateBLP(s::AbstractMatrix, #p::AbstractMatrix,
                     x::AbstractArray{T, 3} where T, ν::AbstractArray{T,3} where T,
                     ivdemand::AbstractArray{T,3} where T,
                     w::AbstractArray{T, 3} where T, ivsupply::AbstractArray{T, 3} where T;
                     method=:MPEC, verbose=true, firmid=1:size(s,1), W=I)

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
  m, ω = supplymoments(γ0, β0, 0*σ0, ξ, x, ν, w, ivsupply, firmid=firmid.*fill(1,T)')
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
        ms, ω = supplymoments(γ, β, σ, ξ, px, ν, w, zs, firmid=firmid.*fill(1,T)')
        m = md[:] + ms[:]
        return(T*m'*W*m)
      end
    end
    @show objectiveBLP(θ0)
    opt = optimize(objectiveBLP, θ0, method=LBFGS(),show_trace=verbose, autodiff=:forward)
    β, σ, γ = unpack(opt.minimizer)
    m, ξ = demandmoments(β,σ, s, x, ν, ivdemand)
    m, ω = supplymoments(γ, β, σ, ξ, x, ν, w, ivsupply, firmid=firmid.*fill(1,T)')
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

    Md = size(ivdemand,1)
    @expression(mod, md[m in 1:Md],
                sum( (ξ[j,t]*ivdemand[m, j, t]) for t in 1:T, j in 1:J)/(J*T))
    Ms=size(ivsupply,1)
    @expression(mod, ms[m in 1:Ms],
                sum(ω[j,t]*ivsupply[m,j,t] for t in 1:T, j in 1:J)/(J*T))
    @assert Md==Ms
    M = Md #+ Ms
    @expression(mod, moments[m in 1:M], md[m] + ms[m]) #m <= Md ? md[m] : ms[m-Md])
    @objective(mod, Min, T*moments'*W*moments);
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
  elseif method==:GEL
    mod = Model()
    K,J,T = size(x)
    Kw = size(w,1)
    S = size(ν,2)
    @variable(mod, β[1:K]) 
    @variable(mod, σ[1:K] ≥ 0)
    @variable(mod, γ[1:Kw])
    @variable(mod, ξ[1:J,1:T])
    @variable(mod, ω[1:J,1:T])
    @variable(mod, p[1:T] ≥ 0)
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

    @assert size(ivdemand,1) == size(ivsupply,1)
    M = size(ivdemand,1)
    @constraint(mod, moments[m in 1:M],
                0 == dot(p, sum(ξ[j,:].*ivdemand[m,j,:] + ω[j,:].*ivsupply[m,j,:]
                                for j in 1:J) ))
    @NLobjective(mod, Max, sum(log(p[t]) for t in 1:T))
    @constraint(mod, sum(p[t] for t in 1:T) <= 1)    
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
           ξ=value.(mod[:ξ]), ω=value.(mod[:ω]), p=value.(mod[:p]),
           opt=mod)
  else
    error("method $method not recognized")
  end
  return(out)    
end


function varianceRCIVlogit(β, σ, s::AbstractMatrix,
                           x::AbstractArray{T, 3} where T, ν::AbstractArray{T,3} where T,
                           ivdemand::AbstractArray{T,3} where T ;
                           firmid=1:size(s,1), W=I)  
  md, ξ = demandmoments(β,σ, s, x, ν, ivdemand)

  J, T = size(ξ)
  Md = size(ivdemand,1)
  mi = similar(ξ, Md, J, T)
  for l in 1:Md
    mi[l, :, :] .= ξ.*ivdemand[l,:,:]
  end
  mi = reshape(mi, size(mi,1)*size(mi,2), size(mi,3))
  V = cov(mi, dims=2)
  θ, unpack = pack(β,σ)
  G = let s=s, x=x, ν=ν, ivdemand=ivdemand
    function(θ)
      β, σ, γ = unpack(θ)
      md, ξ = demandmoments(β,σ, s, x, ν, ivdemand)
      md[:]
    end
  end
  D = ForwardDiff.jacobian(G, θ)
  Ju = ForwardDiff.jacobian(θ->vcat(unpack(θ)...), θ)
  Σ = Ju'*inv(D'*W*D)*(D'*W*V*W*D)*inv(D'*W*D)*Ju/T
  return(Σ=Σ, varm=V)
end

"""
   varianceBLP(β, σ, γ, s::AbstractMatrix,
                     x::AbstractArray{T, 3} where T, ν::AbstractArray{T,3} where T,
                     ivdemand::AbstractArray{T,3} where T,
                     w::AbstractArray{T, 3} where T, ivsupply::AbstractArray{T, 3} where T;
                     firmid=1:size(s,1), W=I)  

Computes variance of BLP estimates. Computes moment variance clustering on `t`. 

Returns `Σ` = covariance of `[β, σ, γ]` and `varm` = (clustered) covariance of moments.  
"""
function varianceBLP(β, σ, γ, s::AbstractMatrix,
                     x::AbstractArray{T, 3} where T, ν::AbstractArray{T,3} where T,
                     ivdemand::AbstractArray{T,3} where T,
                     w::AbstractArray{T, 3} where T, ivsupply::AbstractArray{T, 3} where T;
                     firmid=1:size(s,1), W=I)  
  md, ξ = demandmoments(β,σ, s, x, ν, ivdemand)
  J, T = size(ξ)
  ms, ω = supplymoments(γ, β, σ, ξ, x, ν, w, ivsupply, firmid=firmid.*fill(1, T)')
  
  Md = size(ivdemand,1)
  Ms = size(ivsupply,1)  
  mi = similar(ξ, Md , J, T)
  @assert Md==Ms
  for l in 1:Md
    mi[l, :, :] .= ξ.*ivdemand[l,:,:] .+ ω.*ivsupply[l,:,:]
  end
  #for l in 1:Ms
  #mi[Md+l, :, :] .= ω.*ivsupply[l,:,:]
  #end
  mi = reshape(mi, size(mi,1), size(mi,2), size(mi,3))
  V = cov(reshape(sum(mi, dims=2)/J, size(mi,1), size(mi,3)), dims=2)
  θ, unpack = pack(β,σ,γ)
  G = let s=s, x=x, ν=ν, ivdemand=ivdemand, ivsupply=ivsupply, W=W
    function(θ)
      β, σ, γ = unpack(θ)
      md, ξ = demandmoments(β,σ, s, x, ν, ivdemand)
      ms, ω = supplymoments(γ, β, σ, ξ, x, ν, w, ivsupply, firmid=firmid.*fill(1,T)')
      md[:] + ms[:]
    end
  end
  D = ForwardDiff.jacobian(G, θ)
  Ju = ForwardDiff.jacobian(θ->vcat(unpack(θ)...), θ)
  Σ = Ju'*inv(D'*W*D)*(D'*W*V*W*D)*inv(D'*W*D)*Ju/T
  return(Σ=Σ, varm=V)
end

"""

      polyreg(xpred::AbstractMatrix,
              xdata::AbstractMatrix,
              ydata::AbstractMatrix; degree=1)

Computes  polynomial regression of ydata on xdata. Returns predicted
y at x=xpred. 

# Arguments

- `xpred` x values to compute fitted y
- `xdata` observed x
- `ydata` observed y, must have `size(y)[1] == size(xdata)[1]`
- `degree`
- `deriv` whether to also return df(xpred). Only implemented when
   xdata is one dimentional

# Returns

- Estimates of `f(xpred)`
"""
function polyreg(xpred::AbstractMatrix,
                 xdata::AbstractMatrix,
                 ydata::AbstractMatrix;
                 degree=1, deriv=false)
  function makepolyx(xdata, degree, deriv=false)
    X = ones(size(xdata,1),1)
    dX = nothing
    for d in 1:degree
      Xnew = Array{eltype(xdata), 2}(undef, size(xdata,1), size(X,2)*size(xdata,2))
      k = 1
      for c in 1:size(xdata,2)
        for j in 1:size(X,2)
          @views Xnew[:, k] = X[:,j] .* xdata[:,c]
          k += 1
        end
      end
      X = hcat(X,Xnew)
    end
    if (deriv)
      if (size(xdata,2) > 1)
        error("polyreg only supports derivatives for one dimension x")
      end
      dX = zeros(eltype(X), size(X))
      for c in 2:size(X,2)
        dX[:,c] = (c-1)*X[:,c-1]
      end
    end
    return(X, dX)
  end
  X = makepolyx(xdata,degree)
  (Xp, dXp) = makepolyx(xpred,degree, deriv)
  coef = (X \ ydata)
  ypred = Xp * coef
  if (deriv)
    dy = dXp * coef
    return(ypred, dy)
  else
    return(ypred)
  end
end

"""
    optimalIV(β,σ, γ, 
              s::AbstractMatrix,
              x::AbstractArray{T, 3} where T, ν::AbstractArray{T,3} where T,
              z::AbstractArray{T,3} where T, w::AbstractArray{T, 3} where T;
              firmid=1:size(s,1), degree=2)

Computes optimal instruments for BLP model. Given initial estimates θ=(`β`, `σ`, `γ`), computes
`e(θ)ⱼₜ = (ξ(θ)ⱼₜ , ω(θ)ⱼₜ)`, and then
approximates the optimal instruments with a polynomial regression of `degree` of 
`∂e/∂θ` on `z`. Returns fitted values `(zd, zs) = (E[∂ξ/∂θ|z], E[∂ω/∂θ|z])`  
"""
function optimalIV(β,σ, γ, 
                   s::AbstractMatrix,
                   x::AbstractArray{T, 3} where T, ν::AbstractArray{T,3} where T,
                   z::AbstractArray{T,3} where T, w::AbstractArray{T, 3} where T;
                   firmid=1:size(s,1), degree=2)
  θ, unpack = pack(β,σ, γ)
  J , T = size(s)
  ei = function(θ)
    β, σ, γ = unpack(θ)
    md, ξ = demandmoments(β,σ, s, x, ν, z)
    ms, ω = supplymoments(γ, β, σ, ξ, x, ν, w, z, firmid=firmid.*fill(1, size(ξ,2))')
    vcat(ξ, ω)
  end
  e = ei(θ)
  Ω = cov(e, dims=2)
  Di = reshape(ForwardDiff.jacobian(ei, θ), size(e)..., length(θ))
  Y = zeros(size(e,1)*length(θ), T)
  for t in 1:T
    Y[:,t] .= (inv(Ω)*Di[:,t,:])[:]
  end
  Z = reshape(z, size(z,1)*size(z,2), T)
  zstar = reshape(polyreg(Z',Z',Y', degree=degree)', size(Di,1), size(Di,3),T)
  zd = zeros(size(zstar,2), J, T)
  zs = zeros(size(zstar,2), J, T)
  for j in 1:J
    for t in 1:T
      zd[:,j,t] .= zstar[j, :, t]
      zs[:,j,t] .= zstar[j+J,:,t]
    end
  end
  return((zd, zs))
end
