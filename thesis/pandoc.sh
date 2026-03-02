pandoc main.md \
  --from markdown+tex_math_dollars+tex_math_single_backslash \
  --to docx \
  --reference-doc=reference.docx \
  --citeproc \
  --output=paper.docx