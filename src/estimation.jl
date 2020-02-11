"""
    demandmoments(β::AbstractVector,σ::AbstractVector,
                           dat::BLPData)

Demand side moments in BLP model. 

# Arguments

- `β` length `K` (=number characteristics) vector of average tastes for characteristics
- `σ` length `K` vector of standard deviations of tastes
- `dat::BLPData` 

Returns `(moments, ξ)` where vector of moments is length `L` with `moments[l] = 1/(JT) ∑ⱼ∑ₜ ξ[t][j]*dat[t].zd[l,j]`
  
See also: [`share`](@ref), [`delta`](@ref), [`simulateRCIVlogit`](@ref)
"""
function demandmoments(β::AbstractVector,σ::AbstractVector,
                       dat::BLPData)
  T = length(dat)
  M = size(dat[1].zd,1)
  ξtype = promote_type(eltype(dat[1].x), eltype(β))
  ξ = Array{Array{ξtype, 1}, 1}(undef,T)
  moments = zeros(ξtype, M)
  JT = 0
  for t in 1:T
    ξ[t] = delta(dat[t].s, dat[t].x, dat[t].ν, σ) - dat[t].x'*β
    JT += length(ξ[t])
    moments .+= dat[t].zd*ξ[t]
  end
  moments /= T

  return(moments=moments, ξ=ξ)
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
    supplymoments(γ::AbstractVector, β::AbstractVector, σ::AbstractVector,
                       ξ::AbstractVector, dat::BLPData)


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
- `dat::BLPData` 

Returns `(moments, ω)` an `L` vector of moments with `moments[l] = 1/(T) ∑ₜ∑ⱼ ω[t][j]*dat[t].zs[l,j]` 
"""
function supplymoments(γ::AbstractVector, β::AbstractVector, σ::AbstractVector,
                       ξ::AbstractVector, dat::BLPData)
  T = length(ξ)
  K = length(β)
  M = size(dat[1].zs,1)
  # pre-allocate arrays
  ωtype = promote_type(eltype(ξ[1]), eltype(γ))
  ω = Array{Vector{ωtype},1}(undef, T)
  moments = zeros(ωtype,M)
  JT = 0
  for t in 1:T
    @views p = dat[t].x[1,:]
    @views s, Js, Λ, Γ = dsharedp(β, σ, p, dat[t].x[2:end,:], dat[t].ν, ξ[t])
    Js .= Js .* (dat[t].firmid.==dat[t].firmid')
    mc = p + Js \ s
    @views ω[t] = safelog.(mc) .- dat[t].w'*γ
    moments .+= dat[t].zs*ω[t]
    JT += length(ω[t])
  end
  moments ./= T
  return((moments=moments, ω=ω))
end



"""
    makeivblp(x::AbstractMatrix, firmid=1:size(x,2))

Function for constructing demand instrumental variables following BLP (1995).

Instruments consist of exogenous product characteristics, sum of
rivals exogenous characteristics, and, if there are any multi-product
firms, sum of characteristics of other goods produced by the same
firm.

If includeexp is true, then also use 
`sum(exp(-([x[2:end,j,t] - x[l,j,t])^2) for l in 1:J))`
as instruments. 
"""
function makeivblp(x::AbstractMatrix; firmid=1:size(x,2), includeexp=true)
  K,J = size(x)
  if length(firmid)==length(unique(firmid))
    ivdemand = similar(x, 2*K + K*includeexp, J)
  else
    ivdemand = similar(x, 3*K + K*includeexp, J)    
  end
  ivdemand[1:K,:] .= x
  for j in 1:J
    otherx=x[:,firmid[j] .!= firmid]
    dx = otherx .- x[:,j]
    ivdemand[(K+1):(2K),j] .= vec(sum(otherx, dims=2))
    if (includeexp)
      ivdemand[(2K+1):(3K),j] .= vec(sum( exp.(-(dx).^2), dims=2))
    end
  end
  S = 2*K + K*includeexp
  if length(firmid)!=length(unique(firmid))    
    for j in 1:J
      ivdemand[(S+1):(S+K), j] .= vec(sum(x[:, firmid[j] .== firmid], dims=2)) .- x[:,j]
    end
  end
  
  return(ivdemand)  
end


function makeivblp(x::Array{R, 3}; firmid=(1:size(x,2)) .* fill(1, size(x,3))', includeexp=true) where R
  K,J,T = size(x)
  if size(firmid,1)==length(unique(firmid))
    ivdemand = similar(x, 3*K, J, T)
  else
    ivdemand = similar(x, 4*K, J, T)    
  end
  for t in 1:T
    ivdemand[:,:,t] .= makeivblp(x[:,:,t], firmid=firmid[:,t], includeexp=includeexp)
  end
  return(ivdemand)  
end

""" 
    makeivblp(dat::BLPData; includeexp=true)

Sets dat[:].zd and dat[:].zs to makeivblp([x[2:end,:] w])
"""
function makeivblp(dat::BLPData; includeexp=true)
  out = deepcopy(dat)
  for t in 1:length(dat)
    z = makeivblp(cat(dat[t].x[2:end,:], dat[t].w, dims=1), firmid=dat[t].firmid, includeexp=includeexp)
    out[t] =  MarketData(dat[t].s, dat[t].x, dat[t].w, dat[t].firmid,
                        z, z, dat[t].ν)
  end
  return(out)
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
    estimateRCIVlogit(dat::BLPData; method=:MPEC, verbose=true, W=I)

Estimates a random coefficients IV logit model. 

# Arguments
- `dat::BLPData`
- `method=:MPEC` method for estimation. Available choices are :MPEC, :NFXP, or :GEL
- `verbose=true` whether to display information about optimization progress
- `W=I` weighting matrix
"""
function estimateRCIVlogit(dat::BLPData; method=:MPEC, verbose=true, W=I)
  
  T = length(dat)
  K = size(dat[1].x,1)
  σ0 = ones(K)

  # initial β from logit
  Y = vcat((d->(log.(d.s) .- log(1 .- sum(d.s)))).(dat)...)
  X = hcat( (d->d.x).(dat)...)
  Z = hcat( (d->d.zd).(dat)...)
  xz=(pinv(Z*Z') * Z*X')'*Z
  β0 = (xz*xz') \ xz*Y
  
  if method==:NFXP
    θ0, unpack = pack(β0, σ0)
    objectiveBLP = 
      function(θ)
        β, σ = unpack(θ)
        m, ξ = demandmoments(β,σ, dat)
        return(T*m'*W*m)
      end    
    @show objectiveBLP(θ0)
    opt = optimize(objectiveBLP, θ0, method=LBFGS(),show_trace=verbose, autodiff=:forward)
    β, σ = unpack(opt.minimizer)
    m, ξ = demandmoments(β,σ, dat)
    out = (β=β, σ=σ, ξ=ξ, opt=opt)
  elseif method==:MPEC
    mod = Model()
    K = size(dat[1].x,1)
    Kw = size(dat[1].w,1)
    @variable(mod, β[1:K])
    @variable(mod, σ[1:K] ≥ 0)
    info = VariableInfo(false, NaN, false, NaN, false, NaN, false, NaN, false, false)
    ξ = Vector{Vector{JuMP.variable_type(mod)}}(undef, T)
    JT = 0
    for t in 1:T
      J = length(dat[t].s)
      JT += J
      S = size(dat[t].ν,2)
      ξ[t] = Vector{JuMP.variable_type(mod)}(undef,J)
      for j in 1:J
        ξ[t][j] = JuMP.add_variable(mod, build_variable(error, info), "ξ[$t][$j]")
      end
      njit = @NLexpression(mod, [j in 1:J, i in 1:S], #exp(δi[j,i,t]))
                           exp(sum(dat[t].x[k,j]*β[k] for k in 1:K) + ξ[t][j] +
                               sum(σ[k]*dat[t].ν[k, i]*dat[t].x[k,j] for k in 1:K)))
      dit = @NLexpression(mod, [i in 1:S], 1 + sum(njit[j,i] for j in 1:J))
      sjit = @NLexpression(mod, [j in 1:J, i in 1:S], njit[j,i]/dit[i])
      @NLconstraint(mod, [j in 1:J], dat[t].s[j] == sum(sjit[j,i] for i in 1:S)/S)
    end
    
    Md = size(dat[1].zd,1)
    @expression(mod, moments[m in 1:Md],
                sum( dot(ξ[t],dat[t].zd[m,:]) for t in 1:T)/T)
    @objective(mod, Min, T*moments'*W*moments);
    set_start_value.(mod[:β], β0)
    set_start_value.(mod[:σ], σ0)
    # start from a feasible point
    for t in 1:T
      ξt = delta(dat[t].s, dat[t].x, dat[t].ν, start_value.(mod[:σ])) - dat[t].x'*start_value.(mod[:β])
      set_start_value.(ξ[t], ξt)
    end
    set_optimizer(mod,  with_optimizer(Ipopt.Optimizer,
                                       print_level=5*verbose,
                                       max_iter=1000))
    optimize!(mod)
    out = (β=value.(mod[:β]), σ=value.(mod[:σ]), 
           ξ=nothing, opt=mod)
  elseif method==:GEL
    @warn "method GEL for RCIVlogit might be broken"
    mod = Model()
    K = size(dat[1].x,1)
    @variable(mod, β[1:K])
    @variable(mod, σ[1:K] ≥ 0)
    info = VariableInfo(false, NaN, false, NaN, false, NaN, false, NaN, false, false)
    ξ = Vector{Vector{JuMP.variable_type(mod)}}(undef, T)
    #p = Vector{Vector{JuMP.variable_type(mod)}}(undef, T)
    JT = 0
    for t in 1:T
      J = length(dat[t].s)
      JT += J
      S = size(dat[t].ν,2)
      ξ[t] = Vector{JuMP.variable_type(mod)}(undef,J)
      #p[t] = Vector{JuMP.variable_type(mod)}(undef,J)
      for j in 1:J
        ξ[t][j] = JuMP.add_variable(mod, build_variable(error, info), "ξ[$t][$j]")
        #p[t][j] = JuMP.add_variable(mod, build_variable(error, info), "p[$t][$j]")
        #@constraint(mod, p[t][j] >= 0)
      end
      njit = @NLexpression(mod, [j in 1:J, i in 1:S], #exp(δi[j,i,t]))
                           exp(sum(dat[t].x[k,j]*β[k] for k in 1:K) + ξ[t][j] +
                               sum(σ[k]*dat[t].ν[k, i]*dat[t].x[k,j] for k in 1:K)))
      dit = @NLexpression(mod, [i in 1:S], 1 + sum(njit[j,i] for j in 1:J))
      sjit = @NLexpression(mod, [j in 1:J, i in 1:S], njit[j,i]/dit[i])
      @NLconstraint(mod, [j in 1:J], dat[t].s[j] == sum(sjit[j,i] for i in 1:S)/S)
    end
    
    Md = size(dat[1].zd,1)    
    @variable(mod, p[1:T] ≥ 0)
    M = Md
    @constraint(mod, moments[m in 1:M],
                0 == sum(p[t]*sum(ξ[t][j]*dat[t].zd[m,j] for j in 1:size(dat[t].zd,2)) for t in 1:T))
    #            0 == sum(sum(p[t][j]*ξ[t][j]*dat[t].zd[m,j] 
    #                         for j in 1:size(dat[t].zd,2))
    #                     for t in 1:T))
    #@NLobjective(mod, Max, sum(sum(log(p[t][j]) for j in 1:length(dat[t].s)) for t in 1:T))
    @NLobjective(mod, Max, sum(log(p[t]) for t in 1:T))
    #@constraint(mod, sum(sum(p[t]) for t in 1:T) <= 1)
    @constraint(mod, sum(p) <= 1)
    set_start_value.(mod[:β], β0)
    set_start_value.(mod[:σ], σ0)
    #set_start_value.(mod[:p], 1/T)
    # start from a feasible point
    for t in 1:T
      ξt = delta(dat[t].s, dat[t].x, dat[t].ν, start_value.(mod[:σ])) - dat[t].x'*start_value.(mod[:β])
      set_start_value.(ξ[t], ξt)
      #set_start_value.(p[t], 1/T)
    end
    
    set_optimizer(mod,  with_optimizer(Ipopt.Optimizer,
                                       print_level=5*verbose,
                                       max_iter=1000))
    optimize!(mod)
    out = (β=value.(mod[:β]), σ=value.(mod[:σ]), 
           ξ=nothing, opt=mod)
  else
    error("method $method not recognized")
  end
  return(out)    
end




""" 
    estimateBLP(dat::BLPData; method=:MPEC, verbose=true, W=I)


Estimates a random coefficients BLP demand model

# Arguments
- `dat::BLPData`
- `method=:MPEC` method for estimation. Available choices are :MPEC, :NFXP, or :GEL.
- `verbose=true` whether to display information about optimization progress
- `W=I` `L × L` weighting matrix for moments. 

# Details

Uses `L` unconditional moments for estimation. The moments are
`moments[l] = 1/(T) ∑ₜ∑ⱼ (ξ[t][t]*dat[t].zd[l,j] + ω[t][j]*dat[t].zs[l,j])`

# Methods
- `:NFXP` nested fixed point GMM. `minimize_θ G(δ(θ),θ)'W G(δ(θ),θ)`
- `:MPEC` constrainted GMM. `minimize_{θ,Δ} G(Δ, θ)' W G(Δ, θ) s.t. Δ = δ(θ)`
- `:GEL` constrained GEL `maximize_{p, θ, Δ} ∑ₜ log(p[t]) s.t. E_p[g(Δ, θ)] = 0 and Δ = δ(θ)`

See also: [`optimalIV`](@ref), [`varianceBLP`](@ref), [`simulateBLP`](@ref)
"""
function estimateBLP(dat::BLPData; method=:MPEC, verbose=true, W=I)
  smalls = 1e-4
  if (minimum((d->minimum(d.s)).(dat)) < smalls)
    @warn "There are shares < $smalls."
    @warn "Estimation may encounter numeric problems with small shares."
  end
  if (maximum((d->maximum(d.s)).(dat)) > 1.0 - smalls)
    @warn "There are shares > 1 - $smalls."
    @warn "Estimation may encounter numeric problems with shares near 1."
  end

  T = length(dat)
  K = size(dat[1].x,1)
  σ0 = ones(K)

  # initial β from logit
  Y = vcat((d->(log.(d.s) .- log(1 .- sum(d.s)))).(dat)...)
  X = hcat( (d->d.x).(dat)...)
  Z = hcat( (d->d.zd).(dat)...)
  xz=(pinv(Z*Z') * Z*X')'*Z
  β0 = (xz*xz') \ xz*Y

  # initial γ
  γ0 = zeros(size(dat[1].w, 1))
  m, ξ = demandmoments(β0, 0*σ0, dat)
  m, ω = supplymoments(γ0, β0, 0*σ0, ξ, dat)
  Y = vcat(ω...)
  X = hcat((d->d.w).(dat)...)
  γ0 = X' \ Y  
  @show β0, σ0, γ0
  
  if method==:NFXP
    θ0, unpack = pack(β0, σ0, γ0)
    objectiveBLP = 
      function(θ)
        β, σ, γ = unpack(θ)
        md, ξ = demandmoments(β,σ, dat)
        ms, ω = supplymoments(γ, β, σ, ξ, dat)
        m = md[:] + ms[:]
        return(T*m'*W*m)
      end    
    @show objectiveBLP(θ0)
    opt = optimize(objectiveBLP, θ0, method=LBFGS(),show_trace=verbose, autodiff=:forward)
    β, σ, γ = unpack(opt.minimizer)
    m, ξ = demandmoments(β,σ, dat)
    m, ω = supplymoments(γ, β, σ, ξ, dat)
    out = (β=β, σ=σ, γ=γ, ξ=ξ, ω=ω, opt=opt)
  elseif method==:MPEC
    mod = Model()
    K = size(dat[1].x,1)
    Kw = size(dat[1].w,1)
    @variable(mod, β[1:K])
    @variable(mod, σ[1:K] ≥ 0)
    @variable(mod, γ[1:Kw])
    info = VariableInfo(false, NaN, false, NaN, false, NaN, false, NaN, false, false)
    ξ = Vector{Vector{JuMP.variable_type(mod)}}(undef, T)
    ω = Vector{Vector{JuMP.variable_type(mod)}}(undef, T)
    JT = 0
    for t in 1:T
      J = length(dat[t].s)
      JT += J
      S = size(dat[t].ν,2)
      ξ[t] = Vector{JuMP.variable_type(mod)}(undef,J)
      ω[t] = Vector{JuMP.variable_type(mod)}(undef,J)
      for j in 1:J
        ξ[t][j] = JuMP.add_variable(mod, build_variable(error, info), "ξ[$t][$j]")
        ω[t][j] = JuMP.add_variable(mod, build_variable(error, info), "ω[$t][$j]")
      end
      njit = @NLexpression(mod, [j in 1:J, i in 1:S], #exp(δi[j,i,t]))
                           exp(sum(dat[t].x[k,j]*β[k] for k in 1:K) + ξ[t][j] +
                               sum(σ[k]*dat[t].ν[k, i]*dat[t].x[k,j] for k in 1:K)))
      dit = @NLexpression(mod, [i in 1:S], 1 + sum(njit[j,i] for j in 1:J))
      sjit = @NLexpression(mod, [j in 1:J, i in 1:S], njit[j,i]/dit[i])
      @NLconstraint(mod, [j in 1:J], dat[t].s[j] == sum(sjit[j,i] for i in 1:S)/S)
      Λ = @NLexpression(mod, [j in 1:J], sum(sjit[j,i]*(β[1]+σ[1]*dat[t].ν[1,i]) for i in 1:S)/S)
      Γ = @NLexpression(mod, [j in 1:J, jj in 1:J],
                        (dat[t].firmid[j]==dat[t].firmid[jj])*
                        sum(sjit[j,i]*sjit[jj,i]*(β[1]+σ[1]*dat[t].ν[1,i]) for i in 1:S)/S)
      mc = @NLexpression(mod, [j in 1:J], exp(ω[t][j] + sum(dat[t].w[l,j]*γ[l] for l in 1:Kw)))
      @NLconstraint(mod, [j in 1:J], 0 == dat[t].s[j]/Λ[j] + 
                    dat[t].x[1,j]-mc[j] - sum( (dat[t].x[1,jj] - mc[jj])*Γ[j,jj]/Λ[jj]
                                               for jj in findall(dat[t].firmid[j].==dat[t].firmid)) )
    end
    
    Md = size(dat[1].zd,1)
    @expression(mod, md[m in 1:Md],
                sum( (ξ[t][j]*dat[t].zd[m, j]) for t in 1:T, j in 1:size(dat[t].zd,2))/T)
    Ms=size(dat[1].zs,1)
    @expression(mod, ms[m in 1:Ms],
                sum(ω[t][j]*dat[t].zs[m,j] for t in 1:T, j in 1:size(dat[t].zs,2))/T)
    @assert Md==Ms 
    M = Md #+ Ms
    @expression(mod, moments[m in 1:M], md[m] + ms[m]) #m <= Md ? md[m] : ms[m-Md])
    @objective(mod, Min, T*moments'*W*moments);
    set_start_value.(mod[:β], β0)
    set_start_value.(mod[:σ], σ0)
    set_start_value.(mod[:γ], 0)
    # start from a feasible point
    for t in 1:T
      ξt = delta(dat[t].s, dat[t].x, dat[t].ν, start_value.(mod[:σ])) - dat[t].x'*start_value.(mod[:β])
      set_start_value.(ξ[t], ξt)
      set_start_value.(ω[t],0)
    end
    
    set_optimizer(mod,  with_optimizer(Ipopt.Optimizer,
                                       print_level=5*verbose,
                                       max_iter=1000))
    optimize!(mod)
    out = (β=value.(mod[:β]), σ=value.(mod[:σ]), γ=value.(mod[:γ]),
           ξ=nothing, ω=nothing, opt=mod)
  elseif method==:GEL
    mod = Model()
    K = size(dat[1].x,1)
    Kw = size(dat[1].w,1)
    @variable(mod, β[1:K])
    @variable(mod, σ[1:K] ≥ 0)
    @variable(mod, γ[1:Kw])
    info = VariableInfo(false, NaN, false, NaN, false, NaN, false, NaN, false, false)
    ξ = Vector{Vector{JuMP.variable_type(mod)}}(undef, T)
    ω = Vector{Vector{JuMP.variable_type(mod)}}(undef, T)
    p = Vector{Vector{JuMP.variable_type(mod)}}(undef, T)  
    JT = 0
    for t in 1:T
      J = length(dat[t].s)
      JT += J
      S = size(dat[t].ν,2)
      ξ[t] = Vector{JuMP.variable_type(mod)}(undef,J)
      ω[t] = Vector{JuMP.variable_type(mod)}(undef,J)
      p[t] = Vector{JuMP.variable_type(mod)}(undef,J)
      for j in 1:J
        ξ[t][j] = JuMP.add_variable(mod, build_variable(error, info), "ξ[$t][$j]")
        ω[t][j] = JuMP.add_variable(mod, build_variable(error, info), "ω[$t][$j]")
        p[t][j] = JuMP.add_variable(mod, build_variable(error, info), "p[$t][$j]")
        @constraint(mod, p[t][j] >= 0)
      end
      njit = @NLexpression(mod, [j in 1:J, i in 1:S], #exp(δi[j,i,t]))
                           exp(sum(dat[t].x[k,j]*β[k] for k in 1:K) + ξ[t][j] +
                               sum(σ[k]*dat[t].ν[k, i]*dat[t].x[k,j] for k in 1:K)))
      dit = @NLexpression(mod, [i in 1:S], 1 + sum(njit[j,i] for j in 1:J))
      sjit = @NLexpression(mod, [j in 1:J, i in 1:S], njit[j,i]/dit[i])
      @NLconstraint(mod, [j in 1:J], dat[t].s[j] == sum(sjit[j,i] for i in 1:S)/S)
      Λ = @NLexpression(mod, [j in 1:J], sum(sjit[j,i]*(β[1]+σ[1]*dat[t].ν[1,i]) for i in 1:S)/S)
      Γ = @NLexpression(mod, [j in 1:J, jj in 1:J],
                        (dat[t].firmid[j]==dat[t].firmid[jj])*
                        sum(sjit[j,i]*sjit[jj,i]*(β[1]+σ[1]*dat[t].ν[1,i]) for i in 1:S)/S)
      mc = @NLexpression(mod, [j in 1:J], exp(ω[t][j] + sum(dat[t].w[l,j]*γ[l] for l in 1:Kw)))
      @NLconstraint(mod, [j in 1:J], 0 == dat[t].s[j]/Λ[j] + 
                    dat[t].x[1,j]-mc[j] - sum( (dat[t].x[1,jj] - mc[jj])*Γ[j,jj]/Λ[jj]
                                               for jj in findall(dat[t].firmid[j].==dat[t].firmid)) )
    end
    
    Md = size(dat[1].zd,1)
    Ms=size(dat[1].zs,1)
    @assert Md==Ms
    M = Ms    
    #@variable(mod, p[1:T] ≥ 0)
    @constraint(mod, moments[m in 1:M],
                0 == sum(sum(p[t][j]*(ξ[t][j]*dat[t].zd[m,j] + ω[t][j].*dat[t].zs[m,j])
                             for j in 1:size(dat[t].zd,2))
                         for t in 1:T))
    @NLobjective(mod, Max, sum( sum(log(p[t][j]) for j in 1:length(p[t])) for t in 1:T))
    @constraint(mod, sum(sum(p[t][j] for j in 1:length(p[t])) for t in 1:T) <= 1)    
        set_start_value.(mod[:β], β0)
    set_start_value.(mod[:σ], σ0)
    set_start_value.(mod[:γ], 0)
    #set_start_value.(mod[:p], 1/T)
    # start from a feasible point
    for t in 1:T
      ξt = delta(dat[t].s, dat[t].x, dat[t].ν, start_value.(mod[:σ])) - dat[t].x'*start_value.(mod[:β])
      set_start_value.(ξ[t], ξt)
      set_start_value.(ω[t],0)
      set_start_value.(p[t],1/JT)
    end
    
    set_optimizer(mod,  with_optimizer(Ipopt.Optimizer,
                                       print_level=5*verbose,
                                       max_iter=1000))
    optimize!(mod)
    out = (β=value.(mod[:β]), σ=value.(mod[:σ]), γ=value.(mod[:γ]),
           ξ=nothing, ω=nothing, opt=mod)
  else
    error("method $method not recognized")
  end
  return(out)    
end


"""
   varianceRCIVlogit(β, σ, dat::BLPData ; W=I)  

Computes variance of RCIVlogit estimates. Computes moment variance clustering on `t`. 

Returns `Σ` = covariance of `[β, σ, γ]` and `varm` = (clustered) covariance of moments.  
"""
function varianceRCIVlogit(β, σ, dat::BLPData; W=I)  
  md, ξ = demandmoments(β,σ, dat)

  T = length(dat)
  Md = size(dat[1].zd,1)
  Ms = size(dat[1].zs,1)  
  mi = zeros(eltype(ξ[1]), Md, T)
  @assert Md==Ms
  for t in 1:T
    for l in 1:Ms
      mi[l, t] = dot(ξ[t],dat[t].zd[l,:])
    end
  end
  V = cov(mi, dims=2)
  θ, unpack = pack(β,σ)
  G = function(θ)
    β, σ  = unpack(θ)
    md, ξ = demandmoments(β,σ, dat)
    return(md)
  end
  D = ForwardDiff.jacobian(G, θ)
  Ju = ForwardDiff.jacobian(θ->vcat(unpack(θ)...), θ)
  Σ = Ju'*inv(D'*W*D)*(D'*W*V*W*D)*inv(D'*W*D)*Ju/T
  return(Σ=Σ, varm=V, D=D, Ju=Ju, W=W, mi=mi)
end

"""
   varianceBLP(β, σ, γ, dat::BLPData ; W=I)  

Computes variance of BLP estimates. Computes moment variance clustering on `t`. 

Returns `Σ` = covariance of `[β, σ, γ]` and `varm` = (clustered) covariance of moments.  
"""
function varianceBLP(β, σ, γ, dat::BLPData ; W=I)  
  md, ξ = demandmoments(β,σ, dat)
  ms, ω = supplymoments(γ, β, σ, ξ, dat)

  T = length(dat)
  Md = size(dat[1].zd,1)
  Ms = size(dat[1].zs,1)  
  mi = zeros(eltype(ξ[1]), Md, T)
  @assert Md==Ms
  for t in 1:T
    for l in 1:Ms
      mi[l, t] = (dot(ξ[t],dat[t].zd[l,:]) + dot(ω[t],dat[t].zs[l,:]))
    end
  end
  V = cov(mi, dims=2)
  θ, unpack = pack(β,σ,γ)
  G = function(θ)
    β, σ, γ = unpack(θ)
    md, ξ = demandmoments(β,σ, dat)
    ms, ω = supplymoments(γ, β, σ, ξ, dat)
    md[:] + ms[:]
  end
  D = ForwardDiff.jacobian(G, θ)
  Ju = ForwardDiff.jacobian(θ->vcat(unpack(θ)...), θ)
  Σ = Ju'*pinv(D'*W*D)*(D'*W*V*W*D)*pinv(D'*W*D)*Ju/T
  return(Σ=Σ, varm=V, D=D, Ju=Ju, W=W, mi=mi)
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
    optimalIV(β, σ, γ, dat::BLPData ; degree=2)

Computes optimal instruments for BLP model. Given initial estimates θ=(`β`, `σ`, `γ`), computes
`e(θ)ⱼₜ = (ξ(θ)ⱼₜ , ω(θ)ⱼₜ)`, and then
approximates the optimal instruments with a polynomial regression of `degree` of 
`∂e/∂θ` on `z`. Returns BLPData with instruments set to fitted values `(zd, zs) = (E[∂ξ/∂θ|z], E[∂ω/∂θ|z])`  
"""
function optimalIV(β,σ, γ, 
                   dat::BLPData; degree=2)
  θ, unpack = pack(β,σ, γ)
  ei = function(θ)
    β, σ, γ = unpack(θ)
    md, ξ = demandmoments(β,σ, dat)
    ms, ω = supplymoments(γ, β, σ, ξ, dat)
    hcat(vcat(ξ...), vcat(ω...))
  end
  e = ei(θ)
  Ω = cov(e, dims=1)
  Di = reshape(ForwardDiff.jacobian(ei, θ), size(e)..., length(θ))
  Y = zeros(size(e,1), size(e,2)*length(θ))
  for jt in 1:size(Di,1)
    Y[jt,:] .= (inv(Ω)*Di[jt,:,:])[:]
  end
  Z = hcat((d->[d.zd; d.zs]).(dat)...)'
  zstar = polyreg(Z,Z,Y, degree=degree)
  out = Array{MarketData,1}(undef, length(dat))
  jt = 0
  for t in 1:length(dat)
    J = length(dat[t].s)
    M = size(zstar,2)÷2
    out[t] = MarketData(dat[t].s, dat[t].x, dat[t].w, dat[t].firmid,
                        zstar[jt .+ (1:J), 1:M]',
                        zstar[jt .+ (1:J), (M+1):end]',
                        dat[t].ν)
    jt += J
  end
  @assert jt==size(zstar,1)
  @assert zstar ≈ hcat((d->[d.zd; d.zs]).(out)...)'
  return(out)
end


"""
    fracblp(dat::BLPData)

Estimate random coefficients IV model using "Fast, Robust, Approximately Correct" method of Salanie & Wolak (2019)
"""
function fracRCIVlogit(dat::BLPData)

  T = length(dat)
  K = size(dat[1].x,1)

  # initial β from logit
  Y = vcat((d->(log.(d.s) .- log(1 .- sum(d.s)))).(dat)...)
  X = hcat( (d->d.x).(dat)...)
  V = similar(X) # Salanie & Wolak's K
  JT = 0
  for t in 1:T
    v = similar(dat[t].x)
    for j in 1:size(dat[t].x,2)
      v[:,j] .= (dat[t].x[:,j]./2 - dat[t].x*dat[t].s).*dat[t].x[:,j]
    end
    V[:, (JT+1):(JT+size(v,2))] .= v
    JT = JT + size(v,2)
  end
  XK = vcat(X,V)    
  Z = hcat( (d->d.zd).(dat)...)
  xz=((Z*Z') \ Z*XK')'*Z
  B = (xz*xz') \ xz*Y
  
  return(β=B[1:K], σ=B[(K+1):end])
end
