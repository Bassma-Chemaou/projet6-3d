# Projet 6 - Génération et reconstruction 3D

Deux parties indépendantes réunies dans une même démonstration :

- **Partie A** : génération de formes 3D (nuages de points et voxels) avec des VAE entraînés sur ModelNet10 (chair, table, sofa, bed).
- **Partie B** : reconstruction par Gaussian Splatting d'un objet de maquillage photographié au smartphone, puis compression du modèle et mesure de vitesse.

Contrainte : tout doit tourner sous **12 Go de VRAM** (Google Colab).

## État d'avancement

| Étape | Contenu | Statut |
|---|---|---|
| 1 | Dépôt, environnement, cadrage | en cours |
| 2 | Données ModelNet + prétraitement (A1-A2) + capture photo (B1) | à faire |
| 3 | Baseline AE + COLMAP (A3, B2) | à faire |
| 4 | VAE points + 3DGS (A4, B3) | à faire |
| 5 | Évaluation + compression (A6, B4) | à faire |
| 6 | Démo, docs, rapport | à faire |

## Installation (Colab)

```python
!git clone https://github.com/<votre-compte>/projet6-3d.git
%cd projet6-3d
!pip install -q -r requirements.txt
from google.colab import drive; drive.mount('/content/drive')
from src.check_env import check_env; check_env()
```

## Structure

```
notebooks/   étapes numérotées (01 à 09)
src/         code réutilisable (data, models, recon, losses, metrics, viz)
configs/     un fichier YAML par expérience
data/        scripts de téléchargement et fiches de données (pas de données lourdes)
results/     figures, tableaux, exemples générés, rendus 3DGS
demo/        application Gradio, vidéo de secours, lien de la visionneuse
report/      rapport final
docs/        licences, fiche du modèle, éthique
```

## Données et checkpoints (hors Git)

Les données brutes, photos, checkpoints (.pth) et fichiers .ply sont hébergés hors de GitHub :

- Checkpoints / .ply : _lien Google Drive ou Hugging Face à ajouter_
- Photos de la scène : _lien à ajouter_

## Résultats

_À compléter au fil des étapes (Tableaux A et B)._

## Versions

_À compléter avec `pip freeze` / version de PyTorch et CUDA une fois l'environnement stable._

## Licences et références

Voir `docs/LICENCES.md`.
