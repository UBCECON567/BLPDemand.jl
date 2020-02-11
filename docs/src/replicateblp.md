[![](https://i.creativecommons.org/l/by-sa/4.0/88x31.png)](http://creativecommons.org/licenses/by-sa/4.0/)

This work is licensed under a [Creative Commons Attribution-ShareAlike
4.0 International
License](http://creativecommons.org/licenses/by-sa/4.0/)

### About this document

This document was created using Weave.jl. The code is available in [on
github](https://github.com/UBCECON567/BLPDemand.jl). The same document
generates both static webpages and a [jupyter
notebook.](replicateblp.ipynb)

Introduction
============

This assigment will attempt to replicate the results of (Berry,
Levinsohn, and Pakes [1995](#ref-berry1995)).

Getting started
===============

Load the package for this assignment from github.

``` julia
using Pkg
#Pkg.activate(".") # If running on  vse.syzygy.ca, you might need to uncomment this command
try
  using BLPDemand
catch
  Pkg.develop(PackageSpec(url="https://github.com/UBCECON567/BLPDemand.jl"))
  using BLPDemand
end
```

Problem 1: load and explore the data
====================================

The data from (Berry, Levinsohn, and Pakes [1995](#ref-berry1995)) is
included in the `BLPDemand.jl` package.

``` julia
df = data_blp1999()

# Optionally, if you like being able to view all your data
if (false)
  using TableView
  if (notusingjupyterorjuno) # this will throw an error. Delete it  or the else branch 
    using Blink
    w = Blink.window()
    body!(w, TableView.showtable(df));
  else 
    TableView.showtable(df);
  end
end
```

Create some tables and figures to explore the data. You may want to
reproduce table I and/or II from (Berry, Levinsohn, and Pakes
[1995](#ref-berry1995)).

``` julia
# using Plots, DataFramesMeta, StatsPlots # (or whatever)
```

Problem 2: Logit Demand
=======================

Part I
------

Reproduce table III of (Berry, Levinsohn, and Pakes
[1995](#ref-berry1995)). First you’ll need to create instruments as in
section 5.1 of the paper. The following code should do it.

``` julia
using DataFrames
# Create instruments (see section 5.1)
exog = [:const, :hpwt, :air, :mpd, :space, :mpg, :trend] # not sure that this is correct set of variables
function makeinstruments(df, exog)
  for z ∈ exog
    zown = String(z).*"_own"
    zother = String(z).*"_other"
    own=by(df, [:year, :firm_id], z=>sum)
    rename!(own, Symbol(z, "_sum") => Symbol(z, "_own"))
    all=by(df, [:year], z=>sum)
    rename!(all, Symbol(z, "_sum") => Symbol(z, "_other"))
    df=join(df, own, on = [:year, :firm_id])
    df=join(df,all, on = [:year])
    df[!,Symbol(z,"_other")] .-= df[!,Symbol(z,"_own")]
    df[!,Symbol(z,"_own")] .-= df[!,z]
  end
  return(df)
end
df=makeinstruments(df, exog);
nothing
```

    nothing

For the regressions, you should modify the following.

``` julia
using FixedEffectModels, RegressionTables
col2 = reg(df, @formula(logit_depvar ~ hpwt + (price ~ const_own + const_other + hpwt_own + hpwt_other)))
regtable(col2)
```

    --------------------------
                  logit_depvar
                  ------------
                           (1)
    --------------------------
    (Intercept)      -7.283***
                       (0.166)
    hpwt              5.735***
                       (0.761)
    price            -0.215***
                       (0.015)
    --------------------------
    Estimator               IV
    --------------------------
    N                    2,217
    R2                  -0.236
    --------------------------

Part II
-------

Calculate the elasticities of demand implied by the logit model. Report
how many own price elasticities have absolute value less than one
(inelastic). Why are inelastic demand estimates undesirable?

For computing the elastiticities, you could adapt the code from the
[BLPDemand.jl
docs](https://ubcecon567.github.io/BLPDemand.jl/dev/simulation/#Calculating-Elasticities-1)

Problem 3: Demand Side Estimation
=================================

Estimate a random coefficients demand model. Report the parameter
estimates and standard errors. Use the functions `estimateRCIVlogit` and
`varianceRCIVlogit` from BLPDemand.jl. Note that the paper has price
enter the model as $\log(y-p)$, but we lack data on $y$, so just have
price enter linearly or log-linearly with a random coefficient.

``` julia
# some more data prep. You might want to change
df[!,:loghpwt] = log.(df[!,:hpwt])
df[!,:logmpg] = log.(df[!,:mpg])
df[!,:logspace] = log.(df[!,:space])
S = 20
xvars = [:price, :const, :hpwt, :air, :mpd, :space]
costvars = [:const, :loghpwt, :air, :logmpg, :logspace]
dat = BLPData(df, :year, :firm_id, :share, xvars, costvars,
              [:const], [:const], # we'll recreate instruments anyway
              randn(length(xvars),S ,length(unique(df[!,:year]))) )
dat = makeivblp(dat, includeexp=false)

using LinearAlgebra: diag, I
# now the estimation, you might want to try different options

#out=estimateRCIVlogit(dat, method=:NFXP, verbose=true, W=I)
#v = varianceRCIVlogit(out.β, out.σ, dat, W=I)
#@show [out.β  out.σ]
#@show sqrt.(diag(v.Σ))

# or make a nicer table as in https://ubcecon567.github.io/BLPDemand.jl/dev/simulation/#Estimation-1
```

Problem 4: Demand and Supply
============================

Estimate the full BLP model. Use the `estimateBLP` and `varianceBLP`
functions. See [the
docs](https://ubcecon567.github.io/BLPDemand.jl/dev/simulation/#Estimation-1)
for example usage. Note that the paper has price enter the model as
$\log(y-p)$, but we lack data on $y$, so just have price enter linearly
or log-linearly with a random coefficient.

Problem 5: Elasticities
=======================

Compute price elasticities (with standard errors) based on your
estimates. You can adapt the code in [the
docs](https://ubcecon567.github.io/BLPDemand.jl/dev/simulation/#Calculating-Elasticities-1).

Problem 6: Merger Simulation
============================

This is a more challenging problem with less guidance. Consider it
optional.

Use your estimates to simulate a merger between firm 11 (Saab, I think)
and firm 19 (GM, I think). You can do this by modifying the
`simulateBLP` function. Report the effect of the merger on prices. Be
sure to include standard errors.

<div id="refs" class="references hanging-indent" markdown="1">

<div id="ref-berry1995" markdown="1">

Berry, Steven, James Levinsohn, and Ariel Pakes. 1995. “Automobile
Prices in Market Equilibrium.” *Econometrica* 63 (4): pp. 841–90.
<http://www.jstor.org/stable/2171802>.

</div>

</div>
