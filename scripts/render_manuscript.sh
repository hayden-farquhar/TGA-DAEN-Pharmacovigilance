#!/usr/bin/env bash
# Render manuscript Markdown files to PDF using Pandoc + XeLaTeX
# Usage: bash scripts/render_manuscript.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MANUSCRIPT_DIR="$PROJECT_DIR/manuscript"
OUTPUT_DIR="$MANUSCRIPT_DIR"

echo "=== Rendering manuscript PDFs ==="
echo "    Source: $MANUSCRIPT_DIR"
echo ""

# Common pandoc flags
COMMON=(
    --pdf-engine=xelatex
    -V mainfont="Times New Roman"
    -V papersize=a4
    -V colorlinks=true
    -V linkcolor=black
    -V urlcolor=blue
    -V toccolor=black
    -f markdown+pipe_tables+grid_tables+multiline_tables+table_captions+implicit_figures+raw_tex
    --resource-path="$MANUSCRIPT_DIR"
    --standalone
)

# ── 1. Main manuscript ──────────────────────────────────────────────
echo "[1/3] Rendering manuscript_draft.pdf ..."
pandoc "$MANUSCRIPT_DIR/manuscript_draft.md" \
    -o "$OUTPUT_DIR/manuscript_draft.pdf" \
    "${COMMON[@]}" \
    -V geometry:margin=1in \
    -V fontsize=11pt \
    -V linestretch=1.5
echo "    Done  manuscript_draft.pdf"

# ── 2. Supplementary materials ──────────────────────────────────────
echo "[2/3] Rendering supplementary_materials.pdf ..."
pandoc "$MANUSCRIPT_DIR/supplementary_materials.md" \
    -o "$OUTPUT_DIR/supplementary_materials.pdf" \
    "${COMMON[@]}" \
    -H "$MANUSCRIPT_DIR/landscape_header.tex" \
    -V geometry:margin=0.8in \
    -V fontsize=10pt \
    -V linestretch=1.15
echo "    Done  supplementary_materials.pdf"

# ── 3. READUS-PV checklist ──────────────────────────────────────────
echo "[3/3] Rendering readus_pv_checklist.pdf ..."
pandoc "$MANUSCRIPT_DIR/readus_pv_checklist.md" \
    -o "$OUTPUT_DIR/readus_pv_checklist.pdf" \
    "${COMMON[@]}" \
    -V geometry:margin=1in \
    -V fontsize=10pt \
    -V linestretch=1.15
echo "    Done  readus_pv_checklist.pdf"

echo ""
echo "=== All PDFs rendered ==="
ls -lh "$OUTPUT_DIR"/*.pdf
