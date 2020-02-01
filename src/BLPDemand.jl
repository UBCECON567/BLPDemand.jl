module BLPDemand

import DataFrames
import CSV
import ForwardDiff
import NLsolve
using LinearAlgebra: dot, norm, Diagonal, inv, I
using Optim
using JuMP, Ipopt
#import StatsBase
#import StatsModels

export 
  data_blp,
  share,
  sharep, dsharedp,
  delta,
  demandmoments,
  supplymoments,
  eqprices,
  simulateIVRClogit,
  simulateBLP,
  pack,
  estimateRCIVlogit,
  makeivblp,
  estimateBLP


include("data.jl")
include("share.jl")
include("estimation.jl")
include("simulate.jl")

end # module
