# Detection De Tumeur Cerebrale Sur IRM

Projet Python de classification binaire sur images IRM cerebrales:
- classe 0: sans tumeur
- classe 1: tumeur

Le projet inclut:
- un notebook pedagogique 
- un script d entrainement avance avec transfer learning
- une application web Gradio pour tester des images localement

## Objectif

Construire un modele robuste pour le depistage binaire, avec une priorite sur la reduction des faux negatifs (cas tumeur rates), tout en gardant une interface simple a utiliser.

## Fonctionnalites Principales

- Chargement de plusieurs datasets Kaggle et fusion des donnees.
- Mapping automatique des structures de dossiers vers des labels binaires.
- Reequilibrage explicite du dataset avant entrainement pour limiter le biais de classe.
- Transfer learning avec MobileNetV2 + fine tuning.
- Calibration automatique du seuil de decision sur l ensemble de validation.
- Export des metriques, matrice de confusion et courbes d entrainement.
- Interface Gradio moderne pour inference image par image.

## Structure Du Projet

- brain_tumor_level1.ipynb: notebook d exploration et baseline.
- train_transfer_learning.py: pipeline d entrainement principal.
- app.py: interface web locale Gradio.
- model_manager.py: sauvegarde/chargement modele et metriques.
- utils.py: fonctions utilitaires (affichages, graphiques, etc.).
- models/: modeles sauvegardes (.keras).
- outputs/: metriques JSON et graphes PNG.

## Pre Requis

- Linux, macOS ou Windows.
- Python 3.10+ recommande.
- Un environnement virtuel Python.
- Connexion internet pour telecharger les datasets Kaggle via kagglehub.

## Installation

1. Cloner le projet:

```bash
git clone <url-du-repo>
cd Projet_Tumeur
```

2. Creer et activer un environnement virtuel:

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Installer les dependances:

```bash
pip install --upgrade pip
pip install tensorflow numpy pillow scikit-learn matplotlib pandas gradio kagglehub
```

## Entrainement Avance

Lancer le script:

```bash
source venv/bin/activate
python3 train_transfer_learning.py
```

Le script execute:

1. Telechargement et fusion de plusieurs datasets.
2. Conversion en labels binaires selon les noms de dossiers.
3. Reequilibrage des classes avec ratio cible configurable.
4. Split train/validation/test stratifie.
5. Entrainement en 2 phases:
	- phase 1: tete de classification
	- phase 2: fine tuning partiel
6. Calibration du seuil sur validation.
7. Evaluation finale au seuil 0.50 et au seuil calibre.
8. Sauvegarde du modele et des metriques.

## Datasets Utilises

Le script charge actuellement plusieurs sources definies dans train_transfer_learning.py.

Exemples de sources integrees:
- sartajbhuvaji/brain-tumor-classification-mri
- masoudnickparvar/brain-tumor-mri-dataset
- praneet0327/brain-tumor-dataset

Important:
- les structures de dossiers varient selon les datasets
- le mapping de labels repose sur des tokens de dossiers (ex: no_tumor, notumor, negative, tumor, glioma, etc.)
- en cas de fort desequilibre, ajuster le ratio de reequilibrage dans train_transfer_learning.py

## Calibration Du Seuil

Le modele ne s arrete pas au seuil fixe 0.50.

Le pipeline:
- calcule un seuil calibre sur validation
- privilegie le rappel de la classe tumeur
- sauvegarde ce seuil dans outputs/transfer_results.json

L application lit ensuite ce seuil automatiquement.

## Lancer L Interface Web

```bash
source venv/bin/activate
python3 app.py
```

Puis ouvrir l URL locale fournie par Gradio (souvent http://127.0.0.1:7860).

Fonctions de l app:
- upload d image IRM
- prediction binaire
- affichage probabilites par classe
- mode sensible pour reduire les faux negatifs

## Fichiers De Sortie

- models/brain_tumor_transfer.keras: modele entraine.
- outputs/transfer_results.json: metriques, seuil calibre, distributions de classes.
- outputs/transfer_training_history.png: courbes d entrainement.

## Depannage Rapide

1. Message CUDA non trouve
- Ce n est pas bloquant.
- TensorFlow bascule sur CPU.

2. Desequilibre de classes tres eleve
- Verifier les comptes par dataset dans les logs.
- Ajuster le ratio de reequilibrage dans train_transfer_learning.py.

3. Resultats trop agressifs sur la classe tumeur
- Verifier le seuil calibre.
- Comparer mode normal vs mode sensible dans l app.

4. Erreur modele introuvable dans l app
- Lancer d abord un entrainement pour generer models/brain_tumor_transfer.keras.

## Auteur
Irfat FEJZULLAHU

Projet realise dans un cadre d apprentissage machine learning applique aux IRM cerebrales.
