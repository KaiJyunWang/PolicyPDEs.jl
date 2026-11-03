# Registering later in Julia General

The package is currently unregistered. Installation by Git URL already works.
Registration adds discovery and version resolution through Julia's default registry;
it does not certify the numerical methods.

## Already prepared

- Standard Julia package layout and a permanent UUID in Project.toml.
- MIT license and package author information.
- Julia compatibility bound and CI on Julia 1.10 and the latest stable Julia,
  on Linux and Windows.
- Test environment, contribution guide, and TagBot workflow.
- Development version 0.1.0-DEV; no release or registration is requested automatically.

## Before the first registration

1. Implement and document a useful solver and at least one working economic example.
2. Choose the final package name and check General's current naming and registration
   guidelines. Keep the package UUID unchanged. Keep module, project, and repository
   names consistent if renaming before registration.
3. Add appropriate compatibility bounds for every external dependency as it is added.
   Ensure runtime dependencies are registered in General.
4. Run `julia --project=. -e 'using Pkg; Pkg.test()'` and require passing CI.
   Add numerical correctness tests as functionality arrives. Consider Aqua.jl
   for package quality checks before the first release.
5. Set Project.toml to `version = "0.1.0"`, document release notes, and commit/push.
6. Install the [Registrator GitHub app](https://github.com/apps/juliaregistrator)
   for this repository, then comment `@JuliaRegistrator register` on that commit
   (or an issue). Alternatively, use Registrator's supported web interface.
7. Follow the registration pull request in General and address its checks.
   Once merged, TagBot can create the version tag and GitHub release.
8. Verify `Pkg.add("PolicyPDEs")` in a clean environment and update the README
   with the registered installation command.

TagBot uses the repository's built-in GITHUB_TOKEN with contents-write permission.
No DOCUMENTER_KEY is required for this scaffold. If documentation deployment or
release-triggered workflows are added, review TagBot's authentication guidance.
Commits modifying workflow files may require additional credentials for automated tagging.

Consult the current upstream requirements when registering:

- [General registry](https://github.com/JuliaRegistries/General#registering-a-package-in-general)
- [RegistryCI automatic merging guidelines](https://juliaregistries.github.io/RegistryCI.jl/stable/guidelines/)
- [Registrator](https://github.com/JuliaRegistries/Registrator.jl#how-to-use)
- [TagBot](https://github.com/JuliaRegistries/TagBot)
