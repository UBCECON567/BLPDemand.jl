using Documenter, BLPDemand

makedocs(;
    modules=[BLPDemand],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/schrimpf/BLPDemand.jl/blob/{commit}{path}#L{line}",
    sitename="BLPDemand.jl",
    authors="Paul Schrimpf <paul.schrimpf@gmail.com>",
    assets=String[],
)
