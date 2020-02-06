# Simulations

Here we perform some monte-carlo simulations to illustrute package
usage and estimator performance. See [`Model`](@ref) for an overview
of the model. The tests directory contains some additional examples.

## Generate Data

```@example sim
using BLPDemand, Statistics, PrettyTables, Printf, Random
K = 3             # number of characteristics
J = 5             # number of products
S = 20            # draws of nu
T = 100           # number of markets
β = ones(K)*2 
β[1] = -1.5       # important for equilibrium that higher prices lower sales
σ = ones(K)*0.2
γ = ones(2)*0.3
Random.seed!(984256)

(x, w, p, s, ν, ξ, ω) = simulateBLP(J,T, β, σ, γ, S, varξ=0.2, varω=0.2)
@show quantile(s[:], [0, 0.05, 0.5, 0.95, 1])
```

Estimation will encounter numerical difficulties if market shares are very
close (within 1e-4) to 0 or 1. The above range should be fine. 

## Instruments

In the simulated data `x[2:end,:,:]` and `w` are uncorrelated with `ξ`
and `ω`, but price, `x[1,:,:]`, is endogenous. The price of the `j`th
firm will depend on the characteristics and costs of all other goods,
so these are available as instruments. [`makeivblp`](@ref) is a
convenience function that constructs the sum of all other firms'
exogenous variables to use as instruments. This is similar to what
Berry, Levinsohn, and Pakes (1995) do. We will see below though that
much more accurate estimates can be obtained by using the optimal
instruments. 

```@example sim
z = makeivblp(cat(x[2:end,:,:], w, dims=1));
size(z)
```

## Estimation

We can now estimate the model. Three methods are available:
GMM with nested-fixed point (`:NFXP`), constrained GMM (`:MPEC`), and
constrained GEL (`:GEL`). See [`estimateBLP`](@ref). 

```@repl sim
@time nfxp = estimateBLP(s, x, ν, z, w, z, method=:NFXP, verbose=false)
@time mpec = estimateBLP(s, x, ν, z, w, z, method=:MPEC,verbose=true);
@time gel = estimateBLP(s, x, ν, z, w, z, method=:GEL,verbose=true);
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
vnfxp = varianceBLP(nfxp.β, nfxp.σ, nfxp.γ, s, x, ν, z, w, z);
vmpec = varianceBLP(mpec.β, mpec.σ, mpec.γ, s, x, ν, z, w, z);
nothing
```


Inference for GEL has not been directly implemented. However, GEL is
first order asymptotically equivalent to efficiently weigthed GMM. In
other words, GEL estimates have the same asymptotic variance as
efficiently weighted GMM. 

```@example sim
v = varianceBLP(gel.β, gel.σ, gel.γ, s, x, ν, z, w, z)
vgel = varianceBLP(gel.β, gel.σ, gel.γ, s, x, ν, z, w,z,W=inv(v.varm))

f(v) = @sprintf("(%.2f)", sqrt(v))
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
zd,zs=optimalIV(gel.β, max.(gel.σ, 0.1), # calculating optimal IV with σ near 0 gives poor peformance
                gel.γ, s, x, ν, z, w);
nothing
```

```@repl sim
@time nfxp = estimateBLP(s, x, ν, zd, w, zs, method=:NFXP, verbose=false)
@time mpec = estimateBLP(s, x, ν, zd, w, zs, method=:MPEC,verbose=false);
@time gel = estimateBLP(s, x, ν, zd, w, zs, method=:GEL,verbose=false);
```

```@example sim
vnfxp = varianceBLP(nfxp.β, nfxp.σ, nfxp.γ, s, x, ν, zd, w, zs)
vmpec = varianceBLP(mpec.β, mpec.σ, mpec.γ, s, x, ν, zd, w, zs)
v = varianceBLP(gel.β, gel.σ, gel.γ, s, x, ν, zd, w, zs)
vgel = varianceBLP(gel.β, gel.σ, gel.γ, s, x, ν, zd, w,zs,W=inv(v.varm))

f(v) = @sprintf("(%.2f)", sqrt(v))
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
function elasticity(β, σ, γ, x, ν, ξ)
  K,J,T = size(x)
  e = zeros(typeof(x[:,1,1]'*β+ξ[1,1]),J,K,J,T)
  for t in 1:T
    @views xt = x[:,:,t]
    st(x) = share(x'*β + ξ[:,t], σ, x, ν[:,:,t])
    ∂s = reshape(ForwardDiff.jacobian(st, xt), J, K, J)
    s = st(xt)
    for k in 1:K
      for j in 1:J
        e[:,k,j,t] .= ∂s[:,k,j]./s .* xt[k,j]
      end
    end
  end
  return e
end

function avg_price_elasticity(β,σ,γ)
  ξ = similar(β, J, T)
  for t in 1:T
    ξ[:,t] .= delta(s[:,t], x[:,:,t], ν[:,:,t], σ) - x[:,:,t]'*β
  end
  e=elasticity(β,σ,γ,x,ν,ξ)
  dropdims(mean(e[:,1,:,:], dims=3), dims=3)
end

price_elasticity = avg_price_elasticity(gel.β, gel.σ, gel.γ)
```

Standard errors of elasticities and other quantities calculated from
estimates can be calculated using the delta method.

```@example sim
D = ForwardDiff.jacobian(θ->avg_price_elasticity(θ[1:K], θ[(K+1):(2K)], θ[(2K+1):end]), [gel.β;gel.σ;gel.γ])
V = D*vgel.Σ*D'
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
