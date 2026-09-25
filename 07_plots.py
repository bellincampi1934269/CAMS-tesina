import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Importiamo le  classi
from dataset_pytorch import LongitudinalDataset, collate_fn
from model_architecture import MultimodalDiseasePredictor

def generate_plots():
    print("Inizializzazione per la generazione dei grafici...")
    
    # 1. Caricamento e Split (Stesso seed = stessi identici dati del test precedente)
    dataset = LongitudinalDataset("static_features.csv", "dynamic_features.csv")
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    
    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    
    model = MultimodalDiseasePredictor()
    
    num_negatives = (dataset.static_df['LABEL'] == 0).sum()
    num_positives = (dataset.static_df['LABEL'] == 1).sum()
    pos_weight = torch.tensor([num_negatives / num_positives], dtype=torch.float32)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    epochs = 15
    train_losses = []
    test_losses = []
    
    print("Addestramento veloce per raccogliere i dati storici (circa 1-2 minuti)...")
    
    for epoch in range(epochs):
        # --- TRAINING ---
        model.train()
        epoch_train_loss = 0.0
        for static_b, dyn_b, mask_b, labels_b in train_loader:
            optimizer.zero_grad()
            logits = model(static_b, dyn_b, mask_b).squeeze()
            if logits.dim() == 0:
                logits, labels_b = logits.unsqueeze(0), labels_b.unsqueeze(0)
            loss = criterion(logits, labels_b)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            
        train_losses.append(epoch_train_loss / len(train_loader))
        
        # --- TEST ---
        model.eval()
        epoch_test_loss = 0.0
        test_labels = []
        test_preds = []
        
        with torch.no_grad():
            for static_b, dyn_b, mask_b, labels_b in test_loader:
                logits = model(static_b, dyn_b, mask_b).squeeze()
                if logits.dim() == 0:
                    logits, labels_b = logits.unsqueeze(0), labels_b.unsqueeze(0)
                loss = criterion(logits, labels_b)
                epoch_test_loss += loss.item()
                
                probs = torch.sigmoid(logits)
                test_labels.extend(labels_b.numpy())
                test_preds.extend(probs.numpy())
                
        test_losses.append(epoch_test_loss / len(test_loader))
        print(f"Completata epoca {epoch+1}/{epochs}")

    # ==========================================
    # GENERAZIONE E SALVATAGGIO GRAFICI
    # ==========================================
    print("\nGenerazione grafici in corso...")
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. GRAFICO DELLA LOSS (Learning Curve)
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, epochs+1), train_losses, label='Train Loss', color='blue', linewidth=2)
    plt.plot(range(1, epochs+1), test_losses, label='Test Loss', color='red', linewidth=2, linestyle='--')
    plt.xlabel('Epoca')
    plt.ylabel('Loss (BCEWithLogits)')
    plt.title('Curva di Apprendimento (Train vs Test)')
    plt.legend()
    plt.savefig('plot_01_learning_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. CURVA ROC
    fpr, tpr, thresholds = roc_curve(test_labels, test_preds)
    roc_auc = roc_auc_score(test_labels, test_preds)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Linea del classificatore casuale
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig('plot_02_roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. MATRICE DI CONFUSIONE
    # Convertiamo le probabilità in predizioni nette (0 o 1) con soglia 0.5
    preds_binary = [1 if p > 0.5 else 0 for p in test_preds]
    cm = confusion_matrix(test_labels, preds_binary)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Sano', 'Diabetico'], 
                yticklabels=['Sano', 'Diabetico'])
    plt.xlabel('Predizione del Modello')
    plt.ylabel('Verità (Ground Truth)')
    plt.title('Matrice di Confusione sul Test Set')
    plt.savefig('plot_03_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\nFinito! Controlla la tua cartella, troverai tre file immagine '.png' di alta qualità pronti per il report.")

if __name__ == "__main__":
    generate_plots()
