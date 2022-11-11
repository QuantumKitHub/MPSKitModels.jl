using Documenter
using MPSKitModels

makedocs(;
         modules = [MPSKitModels],
         sitename = "MPSKitModels.jl",
         authors = "Maarten Vandamme",
         format = Documenter.HTML(;
                                  prettyurls = get(ENV, "CI", nothing) == "true",
                                  mathengine = MathJax()),
         pages = [
             "Home" => "home.md",
             "Manual" => [
                 "man/operators.md",
                 "man/mpoham.md",
                 "man/lattices.md",
                 "man/models.md",
             ],
             "Index" => "index.md",
         ])

deploydocs(repo = "github.com/maartenvd/MPSKitModels.jl.git")
