# BLPDemand.jl

Estimate random coefficients demand models in the style of Berry,
Levinsohn, and Pakes (1995).

```@contents
```

# Model

## Demand 

There are $J$ products available. We have data from $T$ markets on
product shares, `s[j,t]`, `K` product characteristics, `x[k,j,t]`, `L`
instruments, `z[l,j,t]`, and `C` cost shifters, `w[c,j,t]`.

Market shares come from a random coefficients model. 
```math
s[j,t] = \int \frac{\exp(x[:,j,t]'(\beta + \nu.*\sigma) + \xi[j,t])}{1 + \sum_{\ell} \exp(x[:,\ell,t]'(\beta + \nu .* \sigma) + \xi[\ell,t])} dF_\nu
```
where ``\beta`` and ``\sigma`` are parameters to be estimates, ``\xi``
are market level demand shocks, and ``\nu`` represents heteregeneity
in tastes for characteristics. Let 
```math
\delta[j,t] = x[:,j,t]'\beta + \xi[j,t]
```
Then we can write shares as
```math
s[j,t] = \int \frac{\exp(\delta[j,t] + x[:,j,t]'(\nu .* \sigma)}{1 + \sum_{\ell} \exp(\delta[\ell,t]' + x[:,\ell,t]'*\nu .* \sigma)} dF_\nu
```
the right hand side of this equation is computed by 
[`share(δ, σ, x,ν)`](@ref). 

Conversely, given `s[:,t]` we can solve for `δ[:,t]` using [`delta(s,
x, ν, σ)`](@ref). 

To estimate ``\theta=(\beta,\sigma)``, we assume that 
```math
E[\xi_{j,t} * z_{\cdot,j,t} ]= 0
```
and minimize a quadratic form in the corresponding empirical moments, 
```julia
m = [mean(xi.*z[l,:,:]) for l in 1:L]
m'*W*m
```
See [`demandmoments`](@ref) and [`estimateRCIVlogit`](@ref). 

## Supply 

For the supply side of the model, we assume that `x[1,:,:]` is
price. Marginal costs are log linear and firms choose prices in
Bertrand-Nash competition, so (for single product firms)
```math
(p[j,t] - exp(w[:,j,t]'\gamma + \omega[j,t])) \frac{\partial
s}{\partial x[1,:,:]} + s[j,t] = 0
```

For estimation, as with demand, we assume that
```math
E[\omega_{j,t} * z_{\cdot,j,t} ]= 0
```
and minimize a quadratic form in the corresponding empirical moments
(along with the demand moments above). 

See [`supplymoments`](@ref) and [`estimateBLP`](@ref)


