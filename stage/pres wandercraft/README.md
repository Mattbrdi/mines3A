# XYZ-IBD — présentation concise

Présentation Beamer en français, en sept diapositives, sans page de titre séparée ni annexes.

## Compilation

Importer le ZIP dans Overleaf, choisir **XeLaTeX** comme compilateur et définir
`presentation_xyz_ibd.tex` comme document principal.

En local, avec une distribution TeX Live récente :

```bash
latexmk -xelatex presentation_xyz_ibd.tex
```

Ou lancer deux fois `xelatex presentation_xyz_ibd.tex`.
Conserver le dossier `figures/` à côté du fichier `.tex`.
Les polices utilisées sont Latin Modern Sans et Latin Modern Math.

## Contenu

1. Introduction : comparaison des datasets et scènes industrielles.
2. Hardware : robot et trois capteurs.
3. Data acquisition pipeline : calibration et capture en deux passes.
4. Annotation error quantification : récupération du bruit, pipeline simulé et résultats.
5. Benchmark : répartition du dataset et quatre tâches.
6. Critères d'évaluation : IoU, AP, MSSD et MSPD.
7. Résultats et conclusion.

## Figures et références

Les cinq images proviennent de la version de l'article fournie par l'utilisateur :
Huang et al., *XYZ-IBD: Benchmarking Robust 6D Object Pose Estimation under
Real-World Industrial Complexity*.

- `dataset_comparaison.png` : figure 4, comparaison de la densité d'instances.
- `scenes.png` : figure 1, scènes du dataset.
- `robot.png` : figure 5, dispositif matériel.
- `acquisition.png` : partie supérieure de la figure 6.
- `spray.png` : figure 7, effet du revêtement antireflet.

La dernière diapositive présente une sélection du tableau 4, avec ses valeurs d'AP.
La validation de la précision des annotations est une mesure en simulation.

Les formules MSSD et MSPD suivent la convention BOP : une symétrie globale
pour tout l'objet, soit `min_S max_x`, avec la transformation de symétrie dans
le repère du modèle. L'article fourni imprime un ordre différent.

- Hodaň et al., *BOP Challenge 2020*, §2.2, équations (2) et (3) :
  https://arxiv.org/abs/2009.07378
- Implémentation BOP, fonctions `mssd` et `mspd` :
  https://github.com/thodan/bop_toolkit/blob/master/bop_toolkit/pose_error.py

Dans l'évaluation COCO, l'AP à seuil IoU fixé repose sur les prédictions triées
par confiance et sur les précisions interpolées à 101 niveaux de rappel.
Le score final moyenne ensuite les AP sur les seuils IoU et les catégories.
Ces détails restent hors des diapositives pour préserver leur lisibilité.
