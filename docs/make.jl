using MPSTransferMatrix
using Documenter

DocMeta.setdocmeta!(MPSTransferMatrix, :DocTestSetup, :(using MPSTransferMatrix); recursive=true)

makedocs(;
    modules=[MPSTransferMatrix],
    authors="Wei Tang <tangwei@smail.nju.edu.cn> and contributors",
    sitename="MPSTransferMatrix.jl",
    format=Documenter.HTML(;
        canonical="https://tangwei94.github.io/MPSTransferMatrix.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/tangwei94/MPSTransferMatrix.jl",
    devbranch="main",
)
