# -*- coding: utf-8 -*-
"""Convertit Rapport_Projet_Deep_Learning.md en PDF (markdown -> HTML -> PDF)."""
import os, markdown
from xhtml2pdf import pisa

root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
md_path = os.path.join(root, "Rapport_Projet_Deep_Learning.md")
pdf_path = os.path.join(root, "Rapport_Projet_Deep_Learning.pdf")

with open(md_path, encoding="utf-8") as f:
    text = f.read()

body = markdown.markdown(
    text, extensions=["tables", "fenced_code", "toc", "sane_lists"])

CSS = """
@page { size: a4 portrait; margin: 2cm 1.8cm; }
body { font-family: Helvetica, Arial, sans-serif; font-size: 10.5pt;
       line-height: 1.45; color: #1a1a1a; }
h1 { font-size: 19pt; color: #14304f; border-bottom: 2px solid #14304f;
     padding-bottom: 4px; margin-top: 18px; }
h2 { font-size: 14pt; color: #14304f; margin-top: 16px;
     border-bottom: 1px solid #c9d4df; padding-bottom: 2px; }
h3 { font-size: 11.5pt; color: #2c3e50; margin-top: 12px; }
p { margin: 5px 0; text-align: justify; }
code { font-family: Courier, monospace; background: #f2f4f6; font-size: 9pt; }
pre { background: #f2f4f6; padding: 6px; font-size: 9pt; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 9.5pt; }
th { background: #14304f; color: #ffffff; border: 1px solid #14304f;
     padding: 4px 6px; text-align: left; }
td { border: 1px solid #b8c2cc; padding: 4px 6px; }
tr:nth-child(even) td { background: #f4f7fa; }
img { width: 15cm; margin: 6px 0; }
blockquote { background: #f4f7fa; border-left: 3px solid #14304f;
             padding: 4px 10px; color: #333; font-style: italic; }
"""

html = f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"

def link_callback(uri, rel):
    # Résout les chemins relatifs des images (figures/...) en chemins absolus.
    path = os.path.join(root, uri.replace("/", os.sep))
    return path if os.path.isfile(path) else uri

with open(pdf_path, "wb") as out:
    result = pisa.CreatePDF(html, dest=out, link_callback=link_callback,
                            encoding="utf-8")

print("Erreurs PDF :", result.err)
print("PDF écrit :", pdf_path, "|", os.path.getsize(pdf_path), "octets")
