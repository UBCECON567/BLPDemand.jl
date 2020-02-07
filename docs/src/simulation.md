# Simulations

Here we perform some monte-carlo simulations to illustrute package
usage and estimator performance. See [`Model`](@ref) for an overview
of the model. The tests directory contains some additional examples.

## Generate Data

```@example sim
using BLPDemand, Statistics, PrettyTables, Printf, Random, LinearAlgebra
K = 3             # number of characteristics
J = 5             # number of products
S = 10            # draws of nu
T = 100           # number of markets
β = ones(K)*2 
β[1] = -1.5       # important for equilibrium that higher prices lower sales
σ = ones(K)
σ[1] = 0.2
γ = ones(2)*0.3
Random.seed!(984256)

(sim, ξ, ω) = simulateBLP(J,T, β, σ, γ, S, varξ=0.2, varω=0.2)
@show quantile(vcat((d->d.s[:]).(sim)...), [0, 0.05, 0.5, 0.95, 1])
```

Estimation will encounter numerical difficulties if market shares are very
close (within 1e-4) to 0 or 1. The above range should be fine. 

## Instruments

In the simulated data `sim[].x[2:end,:]` and `sim[].w` are uncorrelated with `ξ`
and `ω`, but price, `sim[].x[1,:]`, is endogenous. The price of the `j`th
firm will depend on the characteristics and costs of all other goods,
so these are available as instruments. [`makeivblp`](@ref) is a
convenience function that constructs the sum of all other firms'
exogenous variables to use as instruments. This is similar to what
Berry, Levinsohn, and Pakes (1995) do. We will see below though that
much more accurate estimates can be obtained by using the optimal
instruments. `makeivblp` is used by `simulateBLP` to generate `sim[].zd` and `sim[].zs`.

## Estimation

We can now estimate the model. Three methods are available:
GMM with nested-fixed point (`:NFXP`), constrained GMM (`:MPEC`), and
constrained GEL (`:GEL`). See [`estimateBLP`](@ref). 

```@repl sim
@time nfxp = estimateBLP(sim, method=:NFXP, verbose=false)
@time mpec = estimateBLP(sim, method=:MPEC,verbose=true);
@time gel = estimateBLP(sim, method=:GEL,verbose=true);
```

```@example sim
tbl = vcat(["" "True" "NFXP" "MPEC" "GEL"],
           [(i->"β[$i]").(1:length(β)) β nfxp.β mpec.β gel.β],
           [(i->"σ[$i]").(1:length(σ)) σ nfxp.σ mpec.σ gel.σ],
           [(i->"γ[$i]").(1:length(γ)) γ nfxp.γ mpec.γ gel.γ])
pretty_table(tbl, noheader=true, formatter=ft_printf("%6.3f",3:5))
```


In the absence of optimization error and other numeric problems,
`:NFXP` and `:MPEC` should produce identical results.  In finite
sample, we should expect the GEL estimates to differ. All three
methods are consistent, but GEL is also asymptotically
efficient (for a fixed choice of `z`; different `z` can have large
effects on efficiency).

## Inference

[`varianceBLP`](@ref) computes the variance of the estimates produced
by either of GMM estimation methods. 

```@example sim
vnfxp = varianceBLP(nfxp.β, nfxp.σ, nfxp.γ, sim);
vmpec = varianceBLP(mpec.β, mpec.σ, mpec.γ, sim);
nothing
```

Inference for GEL has not been directly implemented. However, GEL is
first order asymptotically equivalent to efficiently weigthed GMM. In
other words, GEL estimates have the same asymptotic variance as
efficiently weighted GMM. 

```@example sim
v = varianceBLP(gel.β, gel.σ, gel.γ, sim)
vgel = varianceBLP(gel.β, gel.σ, gel.γ, sim, W=inv(v.varm))

f(v) = @sprintf("(%.2f)", norm(sqrt(Complex(v))))
vtbl = permutedims(hcat(tbl[1,:],
                        [hcat(tbl[i+1,:],
                              ["", "", f.([vnfxp.Σ[i,i], vmpec.Σ[i,i], vgel.Σ[i,i]])...])
                         for i in 1:size(vgel.Σ,1)]...
                        ))
pretty_table(vtbl, noheader=true, formatter=ft_printf("%6.3f",3:5))
```

As you can see, these standard errors are suspiciously large. It is
possible that there is an error in the code, but I think the problem
lies in the functional form of the unconditional instruments.

## Optimal Instruments

The above estimators use the unconditional moment restriction 
```math
E[(\xi, \omega) z] =0.
```
If we assume the conditional moment restriction,
```math
E[(\xi, \omega) | z] =0.
```
then, we can potentially use any function of ``z`` to form
unconditional moments. 
```math
E[(\xi, \omega) f(z)] =0.
```

The optimal (minimal asymptotic variance) choice of `f(z)` is 
```math
\frac{\partial}{\partial \theta} E[(\xi, \omega) | z] =0.
```

[`optimalIV`](@ref) approximates the optimal instruments by taking as
initial estimate of ``\theta``, computing ``\frac{\partial (\xi,
\omega)}{\partial \theta}`` for each observation in the data, and then
regressing this on a polynomial function of ``z``. Using the fitted
values from this regression as ``f(z)`` results in much more precise
estimates of ``\theta``. 

```@example sim
sim=optimalIV(mpec.β, max.(mpec.σ, 0.1), # calculating optimal IV with σ near 0 gives poor peformance
              mpec.γ, sim, degree=3);
nothing
```

```@repl sim
@time nfxp = estimateBLP(sim, method=:NFXP, verbose=false)
@time mpec = estimateBLP(sim, method=:MPEC,verbose=false);
@time gel = estimateBLP(sim, method=:GEL,verbose=false);
```

```@example sim
vnfxp = varianceBLP(nfxp.β, nfxp.σ, nfxp.γ, sim)
vmpec = varianceBLP(mpec.β, mpec.σ, mpec.γ, sim)
v = varianceBLP(gel.β, gel.σ, gel.γ, sim)
vgel = varianceBLP(gel.β, gel.σ, gel.γ, sim, W=inv(v.varm))

f(v) = @sprintf("(%.2f)", norm(sqrt(Complex(v))))
tbl = vcat(["" "True" "NFXP" "MPEC" "GEL"],
           [(i->"β[$i]").(1:length(β)) β nfxp.β mpec.β gel.β],
           [(i->"σ[$i]").(1:length(σ)) σ nfxp.σ mpec.σ gel.σ],
           [(i->"γ[$i]").(1:length(γ)) γ nfxp.γ mpec.γ gel.γ])
vtbl = permutedims(hcat(tbl[1,:],
                        [hcat(tbl[i+1,:],
                              ["", "", f.([vnfxp.Σ[i,i], vmpec.Σ[i,i], vgel.Σ[i,i]])...])
                         for i in 1:size(vgel.Σ,1)]...
                        ))
pretty_table(vtbl, noheader=true, formatter=ft_printf("%6.3f",3:5))
```

We see that with the optimal instruments all three methods produce
essentially the same results (in this case, theoretically, NFXP=MPEC,
and both are asymptotically equivalent to GEL, so this is
expected). Moreover, the estimates are now much more precise and quite
close to the true parameter values.

## Calculating Elasticities

Using [`share`](@ref) and the `ForwardDiff.jl` package, we can
calculate elasticities of demand with respect to each characteristic.

```@example sim
using ForwardDiff

function elasticity(β, σ, γ, dat, ξ)
  T = length(dat)
  K,J = size(dat[1].x)
  etype = typeof(dat[1].x[:,1]'*β+ξ[1][1])
  e = Array{Array{etype,3},1}(undef, T)
  for t in 1:T
    @views xt = dat[t].x
    st(x) = share(x'*β + ξ[t], σ, x, dat[t].ν)
    ∂s = reshape(ForwardDiff.jacobian(st, xt), J, K, J)
    s = st(xt)
    e[t] = zeros(etype, J,K,J)
    for k in 1:K
      for j in 1:J
        e[t][:,k,j] .= ∂s[:,k,j]./s .* xt[k,j]
      end
    end
  end
  return e
end

function avg_price_elasticity(β,σ,γ, dat)
  ξ = Vector{Vector{eltype(β)}}(undef, T)
  for t in 1:T
    ξ[t] = delta(dat[t].s, dat[t].x, dat[t].ν, σ) - dat[t].x'*β
  end
  e=elasticity(β,σ,γ,dat,ξ)
  avge = zeros(eltype(e[1]),size(e[1]))
  # this assumes J is the same for all markets
  for t in 1:T
    avge .+= e[t]
  end
  avge /= T
  avge[:,1,:]
end

price_elasticity = avg_price_elasticity(mpec.β, mpec.σ, mpec.γ, sim)
```

Standard errors of elasticities and other quantities calculated from
estimates can be calculated using the delta method.

```@example sim
D = ForwardDiff.jacobian(θ->avg_price_elasticity(θ[1:K], θ[(K+1):(2K)], θ[(2K+1):end], dat), [mpec.β;mpec.σ;mpec.γ])
V = D*vmpec.Σ*D'
se = reshape(sqrt.(diag(V)), size(price_elasticity))

tbl = Array{Any, 2}(undef, 2J+1, J+1)
tbl[1,1]=""
for j in 1:J
  tbl[1,j+1] = "price $j"
  tbl[2j,1] = "share $j"
  tbl[2j+1,1] = ""
  for l in 1:J
    tbl[2j, l+1] = price_elasticity[j,l]
    tbl[2j+1, l+1] = @sprintf("(%.3f)",se[j,l])
  end
end
pretty_table(tbl, noheader=true, formatter=ft_printf("%6.3f", 2:(J+1)))       
```

The table above shows the estimated average elasticity of each good's
share with respect to each good's price. Standard errors are in parentheses.
