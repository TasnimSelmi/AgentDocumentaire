# assets/

Ressources visuelles **locales et autorisées uniquement**.

## Logo INSY2S

Aucun logo officiel INSY2S n'est fourni dans ce dépôt. **Ne pas** récupérer
d'asset depuis `https://insy2s.com/` automatiquement.

Pour afficher le logo dans l'en-tête de l'interface, déposer ici un fichier
fourni / validé par l'entreprise, nommé exactement :

- `insy2s-logo.png` (ou `.jpg` / `.jpeg`), **ou**
- `insy2s-logo.svg`

`src/ui/components.py::_render_logo` le détecte automatiquement. Sans fichier,
l'en-tête affiche un simple mot-symbole texte « INSY2S ».

Alternative : variable d'environnement `ADOC_LOGO_PATH` pointant vers le
fichier logo.

## Charte de couleurs

La palette actuelle (`src/ui/styles.py::PROVISIONAL_TOKENS`) est un
**placeholder** — ce ne sont pas les couleurs officielles INSY2S. Pour
appliquer la charte : remplacer les valeurs de `PROVISIONAL_TOKENS` par les
couleurs officielles. Aucun autre fichier à modifier.
