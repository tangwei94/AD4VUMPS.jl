using AD4VUMPS
using Documenter

DocMeta.setdocmeta!(AD4VUMPS, :DocTestSetup, :(using AD4VUMPS); recursive=true)

makedocs(;
    modules=[AD4VUMPS],
    authors="Wei Tang <tangwei@smail.nju.edu.cn> and contributors",
    sitename="AD4VUMPS.jl",
    format=Documenter.HTML(;
        canonical="https://tangwei94.github.io/AD4VUMPS.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/tangwei94/AD4VUMPS.jl",
    devbranch="main",
)
