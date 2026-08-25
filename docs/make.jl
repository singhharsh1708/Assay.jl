using Documenter
using Assay

# Every example in these pages is executed during the build, and the build
# fails if one of them errors. Documentation that is not run is documentation
# that is wrong within a few releases.
DocMeta.setdocmeta!(Assay, :DocTestSetup, :(using Assay); recursive = true)

makedocs(;
    modules = [Assay],
    authors = "Harsh Singh",
    sitename = "Assay.jl",
    format = Documenter.HTML(;
        canonical = "https://singhharsh1708.github.io/Assay.jl",
        edit_link = "main",
        assets = String[],
        size_threshold = 500_000,
    ),
    pages = [
        "Home" => "index.md",
        "Getting started" => "getting-started.md",
        "Declaring a model" => "models.md",
        "Samplers" => "samplers.md",
        "Diagnostics" => "diagnostics.md",
        "Checking that it is right" => "verification.md",
        "Working with other packages" => "interop.md",
        "API reference" => "api.md",
    ],
    checkdocs = :exports,
    warnonly = [:missing_docs],
)

deploydocs(; repo = "github.com/singhharsh1708/Assay.jl", devbranch = "main")
