"""Télécharge ModelNet10 et n'extrait que les catégories choisies (étape A1).

Usage :
    python data/download_modelnet.py --out data/raw --categories chair table sofa bed
"""
import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip"
ALL_CATEGORIES = ["bathtub", "bed", "chair", "desk", "dresser",
                  "monitor", "night_stand", "sofa", "table", "toilet"]


def download(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "ModelNet10.zip"
    if not zip_path.exists():
        print("Téléchargement :", URL)
        urllib.request.urlretrieve(URL, zip_path)
    print("Taille de l'archive (Mo) :", round(zip_path.stat().st_size / 1e6))
    return zip_path


def extract(zip_path: Path, out_dir: Path, categories: list) -> None:
    wanted = tuple(f"ModelNet10/{c}/" for c in categories)
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.namelist() if m.startswith(wanted) and m.endswith(".off")]
        z.extractall(out_dir, members)
    print(len(members), "fichiers .off extraits")


def count(out_dir: Path, categories: list) -> None:
    root = out_dir / "ModelNet10"
    print(f"\n{'categorie':<10}{'train':>7}{'test':>7}")
    for c in categories:
        tr = len(list((root / c / "train").glob("*.off")))
        te = len(list((root / c / "test").glob("*.off")))
        print(f"{c:<10}{tr:>7}{te:>7}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/raw")
    p.add_argument("--categories", nargs="+", default=["chair", "table", "sofa", "bed"])
    a = p.parse_args()
    bad = [c for c in a.categories if c not in ALL_CATEGORIES]
    if bad:
        sys.exit(f"Catégories inconnues : {bad}. Disponibles : {ALL_CATEGORIES}")
    out = Path(a.out)
    extract(download(out), out, a.categories)
    count(out, a.categories)
