using TensorTrainApprox
using Documenter

DocMeta.setdocmeta!(TensorTrainApprox, :DocTestSetup, :(using TensorTrainApprox); recursive=true)

makedocs(;
    modules=[TensorTrainApprox],
    authors="Richard Dennis <richard.dennis@glasgow.ac.uk> and contributors",
    sitename="TensorTrainApprox.jl",
    format=Documenter.HTML(;
        canonical="https://RJDennis.github.io/TensorTrainApprox.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/RJDennis/TensorTrainApprox.jl",
    devbranch="master",
)
