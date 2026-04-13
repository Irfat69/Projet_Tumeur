"""
Utilitaires pour le projet de détection de tumeurs cérébrales.
Fonctions réutilisables pour chargement, prétraitement et évaluation.
"""

import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


def load_images_from_folder(folder_path, img_size=256, label=0):
    images = []
    labels = []
    folder = Path(folder_path)
    
    for img_path in folder.glob("*.jpg") + folder.glob("*.png"):
        try:
            img = Image.open(img_path).convert('L')
            img = img.resize((img_size, img_size))
            images.append(np.array(img) / 255.0)
            labels.append(label)
        except Exception as e:
            print(f"Erreur avec {img_path}: {e}")
    
    return np.array(images), np.array(labels)


def load_dataset(yes_folder, no_folder, img_size=256):
    X_yes, y_yes = load_images_from_folder(yes_folder, img_size=img_size, label=1)
    X_no, y_no = load_images_from_folder(no_folder, img_size=img_size, label=0)
    
    X = np.concatenate([X_yes, X_no], axis=0)
    y = np.concatenate([y_yes, y_no], axis=0)
    
    return X, y


def evaluate_model(model, X_test, y_test, model_name="Model"):
    # Prédictions
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int).flatten()
    
    # Métriques
    acc = accuracy_score(y_test, y_pred_binary)
    prec = precision_score(y_test, y_pred_binary)
    rec = recall_score(y_test, y_pred_binary)
    f1 = f1_score(y_test, y_pred_binary)
    cm = confusion_matrix(y_test, y_pred_binary)
    
    # Affichage
    print(f"\n{'='*50}")
    print(f"Évaluation - {model_name}")
    print(f"{'='*50}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"\nMatrice de confusion:\n{cm}")
    print(f"\nRapport de classification:")
    print(classification_report(y_test, y_pred_binary, 
                               target_names=['No Tumor', 'Tumor']))
    
    return {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist()
    }


def plot_training_history(history, output_path=None):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss pendant l\'entraînement')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy pendant l\'entraînement')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Graphe sauvegardé à: {output_path}")
    
    plt.show()


def display_sample_images(yes_folder, no_folder, n_samples=2):
    yes_folder = Path(yes_folder)
    no_folder = Path(no_folder)
    
    yes_images = list(yes_folder.glob("*.jpg")) + list(yes_folder.glob("*.png"))
    no_images = list(no_folder.glob("*.jpg")) + list(no_folder.glob("*.png"))
    
    fig, axes = plt.subplots(2, n_samples, figsize=(12, 6))
    
    for i in range(n_samples):
        img = Image.open(yes_images[i])
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title("Avec tumeur (yes)")
        axes[0, i].axis('off')
        
        img = Image.open(no_images[i])
        axes[1, i].imshow(img, cmap='gray')
        axes[1, i].set_title("Sans tumeur (no)")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()
