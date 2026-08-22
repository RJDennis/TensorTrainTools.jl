using TensorTrainTools
using Documenter

DocMeta.setdocmeta!(TensorTrainTools, :DocTestSetup, :(using TensorTrainTools); recursive=true)

makedocs(;
    modules=[TensorTrainTools],
    authors="Richard Dennis <richard.dennis@glasgow.ac.uk> and contributors",
    sitename="TensorTrainTools.jl",
    format=Documenter.HTML(;
        canonical="https://RJDennis.github.io/TensorTrainTools.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/RJDennis/TensorTrainTools.jl",
    devbranch="master",
)
