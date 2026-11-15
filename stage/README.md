# XYZ-IBD — présentation Beamer

12 diapositives principales pour environ 10 minutes, suivies de 4 annexes facultatives.

## Fichiers

- `presentation_xyz_ibd.tex` : source éditable de la présentation, avec notes Beamer.
- `presentation_xyz_ibd.pdf` : présentation compilée.
- `notes_orateur.tex`, `.pdf`, `.md` : trame orale et minutage.
- `figures/` : extraits des figures de l’article, nécessaires à la compilation.

## Compilation

Utiliser **XeLaTeX**. Depuis ce dossier :

```sh
latexmk -xelatex presentation_xyz_ibd.tex
latexmk -xelatex notes_orateur.tex
```

Ou lancer deux fois `xelatex` sur chaque fichier. Les polices Latin Modern Sans et Latin Modern Math ainsi que les paquets Beamer, fontspec, unicode-math et PGFPlots sont inclus dans une installation TeX Live complète.

Pour Overleaf : importer cette archive, sélectionner XeLaTeX comme compilateur et `presentation_xyz_ibd.tex` comme document principal. Conserver le dossier `figures/`.

## Périmètre et sources

La présentation suit le résumé personnel « Wandercraft article.pdf » et la version de 21 pages de l’article fournie sous « 1406.pdf ». Les chiffres des autres versions en ligne du projet peuvent différer. Les références sont précisées dans les pieds de page et dans la dernière annexe.

Les illustrations sont des extraits des figures de Huang et al. (figures 1, 3, 5, 6 et 7). Les graphiques sont éditables dans PGFPlots et reprennent exactement le tableau 4. Les critiques méthodologiques et les exemples explicatifs sont identifiés comme tels.
