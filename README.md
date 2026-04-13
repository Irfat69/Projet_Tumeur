# Projet Tumeur Cérébrale

Projet de classification binaire IRM pour détecter la présence d'une tumeur cérébrale.

## Contenu du projet

- `brain_tumor_level1.ipynb` : exploration, prétraitement et entraînement initial
- `train_transfer_learning.py` : entraînement avancé avec transfer learning
- `app.py` : interface web locale pour tester une image IRM
- `model_manager.py` : chargement et sauvegarde du modèle
- `utils.py` : fonctions utilitaires partagées

## Lancer l'interface

```bash
source venv/bin/activate
python app.py
```

## Relancer l'entraînement avancé

```bash
source venv/bin/activate
python train_transfer_learning.py
```

## Sorties générées

- `models/` : modèles sauvegardés
- `outputs/` : métriques et graphiques d'entraînement

## Remarque

Ce projet est pédagogique et ne remplace pas un avis médical.
