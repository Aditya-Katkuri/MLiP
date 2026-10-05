#!/usr/bin/env bash
# Render the pitch deck to PDF with LibreOffice, then each page to PNG for a visual check.
# usage: bash reports/presentation/render.sh
set -euo pipefail
cd "$(dirname "$0")"
soffice --headless --norestore --convert-to pdf --outdir . sidewalk-scanner-pitch.pptx > /dev/null 2>&1
/data/tools/shotenv/bin/python - <<'EOF'
import pypdfium2 as pdfium
pdf = pdfium.PdfDocument("sidewalk-scanner-pitch.pdf")
for i, page in enumerate(pdf):
    page.render(scale=1.6).to_pil().save(f"build/page{i + 1}.png")
print(f"{len(pdf)} pages -> build/page*.png")
EOF
