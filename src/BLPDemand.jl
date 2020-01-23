module BLPDemand

import DataFrames
import CSV
import ForwardDiff
using LinearAlgebra: dot, norm, Diagonal, inv
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
  simulateBLP

include("data.jl")
include("estimation.jl")
include("simulate.jl")

end # module
