# Detection de tumeur cerebrale sur IRM

Prototype de deep learning pour l analyse d images IRM 2D avec deux taches:
1. detection binaire: tumeur vs aucune tumeur
2. classification du type de tumeur: glioma, meningioma, pituitary

Ce projet combine entrainement, evaluation et interface web locale.

> Important: ce projet est un prototype pedagogique. Il ne constitue pas un dispositif medical, ne fournit pas de diagnostic et ne doit jamais remplacer l'avis d'un professionnel de sante.

## Objectifs

1. Construire un modele binaire robuste avec priorite sur la reduction des faux negatifs.
2. Ajouter un modele multiclasses pour proposer un type suspect quand une tumeur est detectee.
3. Fournir une app Gradio simple pour tester des IRM localement.
4. Afficher une carte Grad-CAM pour visualiser les zones influentes de l'image.

## Stack technique

1. Python 3
2. TensorFlow / Keras (CNN par transfer learning)
3. scikit-learn (split, metriques, class weights)
4. Gradio (interface)
5. kagglehub (telechargement datasets)

## Architecture du projet

1. `app.py`
Interface web Gradio. Charge le modele binaire et affiche:
- verdict principal
- repartition des classes
- type suspect (si detecte)
- visualisation Grad-CAM des zones influentes

2. `train_transfer_learning.py`
Pipeline d entrainement du modele binaire.

3. `train_tumor_type_classifier.py`
Pipeline d entrainement multiclasses du type de tumeur.

4. `model_manager.py`
Sauvegarde et chargement des modeles/metriques.

5. `utils.py`
Utilitaires de visualisation et evaluation.

6. `brain_tumor.ipynb`
Notebook d exploration initiale avec un baseline CNN simple.

7. `models/`
Modeles `.keras` sauvegardes.

8. `outputs/`
Metriques JSON et graphes d entrainement.

## Installation

1. Cloner le depot

```bash
git clone <url-du-repo>
cd Projet_Tumeur
```

2. Creer un environnement virtuel

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Installer les dependances

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Datasets utilises

Les sources sont configurees dans `train_transfer_learning.py` et `train_tumor_type_classifier.py`.

Exemples de sources utilises:
1. `sartajbhuvaji/brain-tumor-classification-mri`
2. `masoudnickparvar/brain-tumor-mri-dataset`
3. `ashkhagan/figshare-brain-tumor-dataset` (format `.mat`)

Note:
Le dataset Figshare ne suit pas la structure dossier par classe classique. Les labels sont stockes dans les fichiers `.mat`.

## Entrainement

### 1) Modele binaire

```bash
source venv/bin/activate
python3 train_transfer_learning.py
```

Sorties principales:
1. `models/brain_tumor_transfer.keras`
2. `outputs/transfer_results.json`
3. `outputs/transfer_training_history.png`

### 2) Modele type de tumeur

```bash
source venv/bin/activate
python3 train_tumor_type_classifier.py
```

Sorties principales:
1. `models/brain_tumor_type.keras`
2. `outputs/tumor_type_results.json`
3. `outputs/tumor_type_training_history.png`

## Lancer l application

```bash
source venv/bin/activate
python3 app.py
```

Puis ouvrir l URL locale fournie par Gradio (souvent `http://127.0.0.1:7860`).

Pour utiliser un autre port:

```bash
GRADIO_SERVER_PORT=8501 python3 app.py
```

## Notebook d exploration

Le fichier `brain_tumor.ipynb` est conserve comme trace pedagogique de l exploration initiale:
1. chargement d un petit dataset binaire
2. visualisation de quelques IRM
3. entrainement d un CNN simple
4. evaluation de base

Les pipelines de reference du projet sont les scripts Python:
1. `train_transfer_learning.py` pour le modele binaire final
2. `train_tumor_type_classifier.py` pour le modele multiclasses
3. `app.py` pour l interface Gradio avec Grad-CAM

## Resultats recents

### Detection binaire

Run recent apres deduplication exacte et reequilibrage:
1. Accuracy: 0.9756
2. Precision: 0.9753
3. Recall: 0.9884
4. F1-score: 0.9818
5. Seuil calibre: 0.29

Matrice de confusion:

```text
[[247, 13],
 [  6, 513]]
```

Lecture rapide:
1. le modele garde un rappel eleve sur la classe tumeur
2. 6 faux negatifs sur le jeu de test
3. 8069 doublons exacts retires avant le split

### Classification du type de tumeur

Run recent apres deduplication exacte:
1. Accuracy: 0.8341
2. Precision macro: 0.8347
3. Recall macro: 0.8291
4. F1 macro: 0.8215

Matrice de confusion:

```text
[[470,  54,  35],
 [ 28, 246, 105],
 [  0,   1, 405]]
```

Lecture rapide:
1. 8960 images uniques apres suppression de 2268 doublons exacts
2. split stratifie: 6272 train, 1344 validation, 1344 test
3. classe la plus difficile: meningioma, souvent confondue avec pituitary

## Evaluation et suivi

Le projet exporte:
1. confusion matrix
2. classification report par classe
3. historique entrainement/validation
4. resume JSON pour reproductibilite
5. carte Grad-CAM dans l'interface de prediction

## Limites actuelles

1. projet pedagogique non certifie medicalement
2. risque residuel de biais si des images tres proches mais non identiques se retrouvent a la fois en train et test
3. heterogeneite des sources Kaggle (qualite d annotation variable)
4. absence de validation clinique externe

## Plan d amelioration

1. split plus robuste par groupe patient quand possible
2. detection des doublons visuels proches, au-dela des doublons exacts
3. test de backbones alternatifs (EfficientNet)
4. calibration et strategie de seuil par classe
5. analyse approfondie des erreurs meningioma vs pituitary

## Portfolio: points forts a mettre en avant

1. pipeline end-to-end (data, training, eval, app)
2. approche orientee metriques et reduction des faux negatifs
3. integration de plusieurs sources de donnees
4. audit explicite de la qualite des labels

## Auteur

Irfat FEJZULLAHU
