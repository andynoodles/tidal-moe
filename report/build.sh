#!/usr/bin/env bash
# Rebuilds the technical report PDF. Figures: python3 figures/make_figures.py (needs numpy + matplotlib).
# LaTeX runs in the TeX Live container, so nothing needs to be installed on the host.
#   report/build.sh            -> report/tidalmoe.pdf
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ "${FIGURES:-1}" = 1 ] && python3 -c "import matplotlib, numpy" 2>/dev/null; then python3 figures/make_figures.py; fi
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD":/work -w /work --entrypoint latexmk \
  texlive/texlive:latest-medium -pdf -interaction=nonstopmode -halt-on-error tidalmoe.tex
latexmk_aux=(tidalmoe.aux tidalmoe.bbl tidalmoe.blg tidalmoe.fdb_latexmk tidalmoe.fls tidalmoe.log tidalmoe.out)
rm -f "${latexmk_aux[@]}"
echo "built $(pwd)/tidalmoe.pdf"
