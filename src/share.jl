
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

  s = zeros(promote_type(eltype(δ), eltype(σ)),size(δ))
  si = similar(s)
  σx = σ.*x
  @inbounds for i in 1:S
    @simd for j in 1:J
      @views si[j] = δ[j] + dot(σx[:,j], ν[:,i])      
    end
    # to prevent overflow from exp(δ + ...)
    simax = max(maximum(si), 0)
    si .-= simax    
    si .= exp.(si)
    si .= si./(exp.(-simax) + sum(si))
    s .+= si
  end
  s ./= S
  s .+= eps(zero(eltype(s)))
  #@show s, δ
  return(s)
end

"""
   function sharep(β::AbstractVector,
                   σ::AbstractVector,
                   p::AbstractVector,
                   x::AbstractMatrix,
                   ν::AbstractMatrix,
                   ξ::AbstractVector)

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
                ξ::AbstractVector)
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
function delta(s::AbstractVector{T},
               x::AbstractMatrix{T},
               ν::AbstractMatrix{T},
               σ::AbstractVector{T};
               tol=sqrt(eps(eltype(s))), maxiter=1000) where T

  # δ = log.(s) .- log.(1-sum(s))
  # δold = copy(δ)
  # smodel = copy(s)  
  # normchangeδ = 10*tol
  # normserror = 10*tol
  # iter = 0
  # while (normchangeδ > tol) && (normserror > tol) && iter < maxiter
  #   δ, δold = δold, δ
  #   smodel .= share(δold, σ, x, ν)
  #   smodel .= max.(smodel, eps(0.0)) # avoid log(negative)
  #   δ .= δold .+ log.(s) .- log.(smodel)
  #   normchangeδ = norm(δ - δold)
  #   normserror = norm(s - smodel)
  #   iter += 1
  # end

  # if (iter>maxiter)
  #   @warn "Maximum iterations ($maxiter) reached"
  # end
  # return(s)

  #@show σ

  sol = try
    # Anderson acceleration (this is generally faster, but fails once in a while
    sol=NLsolve.fixedpoint(d->(d .+ log.(s) .- log.(share(d, σ, x, ν))),
                         log.(s) .- log.(1-sum(s)), 
                         method = :anderson, m=5, xtol=tol, ftol=tol,
                         iterations=maxiter, show_trace=false)
    (norm(share(sol.zero, σ, x, ν) - s)<tol) || error("bad sol")
    sol
  catch
    # fixed point iteration, always works, but takes more iterations
    #println("trying fixed point")
    sol = NLsolve.fixedpoint(d->(d .+ log.(s) .- log.(share(d, σ, x, ν))),
                             log.(s) .- log.(1-sum(s)), 
                             method = :anderson, m=0, xtol=tol, ftol=tol,
                             iterations=maxiter, show_trace=false)
    #(norm(share(sol.zero, σ, x, ν) - s)<tol) || error("bad sol")
    sol
  end
  return(sol.zero)
end

function delta(s::AbstractVector, x::AbstractMatrix, ν::AbstractMatrix,
               σ::Vector{D}; kw...) where {D <: ForwardDiff.Dual}
  σval = ForwardDiff.value.(σ)
  δ = delta(s,x,ν, σval, kw...)
  ∂δ = ForwardDiff.jacobian(d -> share(d, σval, x, ν), δ)
  ∂σ = ForwardDiff.jacobian(s -> share(δ, s, x, ν), σval)
  out = similar(σ, length(δ))
  #J = Array{eltype(σ), 2}(undef, length(δ), length(σ))
  Jv = try
    -∂δ \ ∂σ
  catch
    #@show ∂δ, δ, ∂σ
    zeros(eltype(∂σ),size(∂σ))
  end
  Jc = zeros(ForwardDiff.valtype(D), length(σ), ForwardDiff.npartials(D))
  for i in eachindex(σ)
    Jc[i,:] .= ForwardDiff.partials(σ[i])
    #ForwardDiff.extract_jacobian!(D, Jc, σ, ForwardDiff.npartials(D))
  end
  Jn = Jv * Jc
  for i in eachindex(out)
    out[i] = D(δ[i], ForwardDiff.Partials(tuple(Jn[i,:]...)))
  end
  return(out)
end

# @adjoint delta(s,x, ν, σ; kw...) = 
#   let δ = delta(s,x, ν, σ; kw...) 
#     δ, function(vresult)
#       # This backpropagator returns (- v' (ds/dδ)⁻¹ (ds/dp))'
#       v = vresult
#       J = dsharedδ(δ,σ,x,ν)
#       _, back = forward(share->share(δ,σ,x,ν), share)
#       return (back(-(J' \ v))[1], nothing, nothing)
#     end
#   end
