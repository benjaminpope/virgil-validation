#!/usr/bin/env bash
# Build the pinned Julia project for the orbit comparison (plan section 2, "Julia" row; WP-O only).
#   juliaup:  brew install juliaup && juliaup add 1.12 && juliaup default 1.12   (done once, outside this script)
#   project:  orbit/julia/{Project.toml,Manifest.toml}; Octofitter 9.0.0 from the registry, OctofitterInterferometry from
#             the tag v9.0.0 commit of the Octofitter.jl repository (subdir OctofitterInterferometry).
# Preflight: stop if free disk < 9 GB. After install: record sizes and the precompile time; if free < 5 GB run Pkg.gc()
# and continue only if >= 4 GB. Log: ~/.cache/virgil-bench/logs/setup_julia.log
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
LOG="${HOME}/.cache/virgil-bench/logs/setup_julia.log"
mkdir -p "$(dirname "$LOG")"
free_gb() { df -g / | awk 'NR==2 {print $4}'; }
FREE=$(free_gb)
echo "free disk before: ${FREE} GB" | tee -a "$LOG"
if [ "$FREE" -lt 9 ]; then echo "free disk < 9 GB: stop (fallback: orbitize! comparator)" | tee -a "$LOG"; exit 2; fi
OCTO_REV=5649836e82ba2530a6df03d0ee032c05b77f0ceb   # peeled commit of tag v9.0.0 (git ls-remote --tags)
T0=$(date +%s)
julia --project="$HERE/julia" -e '
using Pkg
Pkg.Registry.update()
Pkg.add(name="Octofitter", version=v"9.0.0")
Pkg.add(PackageSpec(url="https://github.com/sefffal/Octofitter.jl", subdir="OctofitterInterferometry", rev="'"$OCTO_REV"'"))
Pkg.add(["Distributions", "Pigeons", "MCMCChains", "CSV", "JSON3", "TypedTables"])
' 2>&1 | tee -a "$LOG"
T1=$(date +%s)
julia --project="$HERE/julia" -e 'using Pkg; Pkg.precompile()' 2>&1 | tee -a "$LOG"
T2=$(date +%s)
{
  echo "resolve+download seconds: $((T1-T0)); precompile seconds: $((T2-T1))"
  echo "sizes: $(du -sh ~/.julia ~/.julia/juliaup 2>/dev/null | tr '\n' ' ')"
  echo "free disk after: $(free_gb) GB"
  julia --version
} | tee -a "$LOG"
if [ "$(free_gb)" -lt 5 ]; then
  julia --project="$HERE/julia" -e 'using Pkg; Pkg.gc()' | tee -a "$LOG"
  echo "free disk after gc: $(free_gb) GB" | tee -a "$LOG"
  [ "$(free_gb)" -ge 4 ] || { echo "below the 4 GB floor: stop" | tee -a "$LOG"; exit 3; }
fi
