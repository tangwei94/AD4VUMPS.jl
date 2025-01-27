using AD4vumps
using Documenter

DocMeta.setdocmeta!(AD4vumps, :DocTestSetup, :(using AD4vumps); recursive=true)

makedocs(;
    modules=[AD4vumps],
    authors="Wei Tang <tangwei@smail.nju.edu.cn> and contributors",
    sitename="AD4vumps.jl",
    format=Documenter.HTML(;
        canonical="https://tangwei94.github.io/AD4vumps.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/tangwei94/AD4vumps.jl",
    devbranch="main",
)
