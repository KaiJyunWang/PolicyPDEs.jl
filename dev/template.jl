# PkgTemplates.jl recipe used to generate the scaffold (PkgTemplates 0.7.63).
# Run in a separate environment with PkgTemplates installed. This creates a NEW
# package at ARGS[1]/PolicyPDEs and refuses to overwrite an existing directory.
using PkgTemplates

destination = isempty(ARGS) ? mktempdir() : abspath(ARGS[1])
template = Template(;
    user = "KaiJyunWang",
    authors = ["Kai-Jyun Wang <b11303072@ntu.edu.tw> and contributors"],
    dir = destination,
    julia = v"1.10",
    plugins = [
        ProjectFile(; version = v"0.1.0-DEV"),
        License(; name = "MIT"),
        Git(; branch = "main", manifest = false),
        GitHubActions(; linux = true, windows = true, osx = false,
                      coverage = false, extra_versions = ["1"]),
        Tests(; project = true),
        TagBot(; ssh = nothing),
    ],
)
template("PolicyPDEs")

# Post-generation edits in this repository add the design and registration docs,
# latest-stable Julia to CI, explicit TagBot permissions, and nested manifest ignores.
