# DATA CARD - Données de génération 3D

## Source
- **Jeu de données** : ModelNet10 (sous-ensemble de ModelNet40), Princeton.
- **URL** : http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip (page : https://modelnet.cs.princeton.edu/download.html)
- **Référence** : Z. Wu, S. Song, A. Khosla, F. Yu, L. Zhang, X. Tang, J. Xiao, *3D ShapeNets: A Deep Representation for Volumetric Shapes*, CVPR 2015.
- **Conditions d'usage** : pas de licence formelle ; le site indique que le jeu est fourni pour la recherche académique uniquement. Les droits des modèles CAO appartiennent à leurs auteurs. Usage strictement académique dans ce projet.
- **Date de téléchargement** : 3 novembre 2026

## Périmètre retenu
- Catégories : chair, table, sofa, bed
- Découpage : train / test officiels de ModelNet10 ; train plafonné à 400 formes par catégorie (garde le total proche de 2 000 objets) ; validation = 15 % du train (tirage stratifié, graine 42)

## Effectifs

| Catégorie | Train | Val | Test |
|---|---|---|---|
| chair | 340 | 60 | 100 |
| table | 333 | 59 | 100 |
| sofa | 340 | 60 | 100 |
| bed | 340 | 60 | 100 |
| **Total** | 1353 | 239 | 400 |

Total : 1 992 objets. Aucune forme ignorée au prétraitement.

## Prétraitement
- Lecture des maillages .off ; normalisation : centrage sur la boîte englobante et mise à l'échelle dans [-1, 1].
- Nuages de points : 2 048 points échantillonnés sur la surface (un sous-ensemble de 1 024 s'obtient en prenant les 1 024 premiers points).
- Voxels : grille d'occupation 32x32x32 (0/1), remplie, stockée compressée.
- Axe vertical : Z.

## Limites
- Catégories de meubles uniquement ; peu de diversité de formes.
- Maillages parfois non étanches ; les voxels sont remplis par fermeture morphologique, ce qui peut épaissir les parties fines.

## Données hors Git
Les fichiers `data/raw/` et `data/processed/` ne sont pas versionnés (voir `.gitignore`). Sauvegarde sur Google Drive : dossier `projet6-3d/processed`.
