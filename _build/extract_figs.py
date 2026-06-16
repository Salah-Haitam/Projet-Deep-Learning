# -*- coding: utf-8 -*-
"""Extrait les figures (image/png) déjà présentes dans les notebooks exécutés."""
import nbformat, base64, os

root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
figdir = os.path.join(root, "figures")
os.makedirs(figdir, exist_ok=True)

def dump(nb_name, prefix):
    nb = nbformat.read(os.path.join(root, nb_name), 4)
    k = 0
    saved = []
    for ci, c in enumerate(nb.cells):
        if c.cell_type != "code":
            continue
        for o in c.get("outputs", []):
            data = o.get("data", {})
            if "image/png" in data:
                png = base64.b64decode(data["image/png"])
                fn = f"{prefix}_{k:02d}.png"
                with open(os.path.join(figdir, fn), "wb") as f:
                    f.write(png)
                saved.append((ci, fn, len(png)))
                k += 1
    print(f"\n=== {nb_name} : {k} figures ===")
    for ci, fn, n in saved:
        print(f"  cellule {ci:2d} -> {fn} ({n} octets)")

dump("Partie1_MLP_WineQuality.ipynb", "p1")
dump("Partie2_CNN_CIFAR10.ipynb", "p2")
dump("Partie3_RNN_Seq2Seq_FraEng.ipynb", "p3")
print("\nDossier figures :", figdir)
