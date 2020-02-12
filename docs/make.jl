using Documenter, BLPDemand

makedocs(;
         modules=[BLPDemand],
         format=Documenter.HTML(),
         pages=[
           "Home" => "index.md",
           "Simulations" => "simulation.md",
           "Function Reference" => "functions.md",
           "Developer Notes" => "implementation.md",
           "Replicating BLP" => "replicateblp.md"
         ],
         repo="https://github.com/UBCECON567/BLPDemand.jl/blob/{commit}{path}#L{line}",
         sitename="BLPDemand.jl",
         authors="Paul Schrimpf <schrimpf@mail.ubc.ca>",
         doctest=true
         #assets=String[],
)


deploydocs(repo="github.com/UBCECON567/BLPDemand.jl.git")
