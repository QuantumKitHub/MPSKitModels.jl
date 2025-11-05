# if examples is not the current active environment, switch to it
if Base.active_project() != joinpath(@__DIR__, "Project.toml")
    using Pkg
    Pkg.activate(@__DIR__)
    Pkg.develop(PackageSpec(; path = (@__DIR__) * "/../"))
    Pkg.resolve()
    Pkg.instantiate()
end

using Documenter
using MPSKitModels

makedocs(;
    modules = [MPSKitModels],
    sitename = "MPSKitModels.jl",
    authors = "Maarten Vandamme",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", nothing) == "true",
        mathengine = MathJax()
    ),
    pages = [
        "Home" => "index.md",
        "Manual" => [
            "man/operators.md",
            "man/mpoham.md",
            "man/lattices.md",
            "man/models.md",
        ],
        "Index" => "package_index.md",
    ],
    checkdocs = :public
)

deploydocs(; repo = "github.com/QuantumKitHub/MPSKitModels.jl.git")
