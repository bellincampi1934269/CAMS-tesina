import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score, classification_report
import numpy as np

# Importiamo le classi
from dataset_pytorch import LongitudinalDataset, collate_fn
from model_architecture import MultimodalDiseasePredictor

def train_and_evaluate():
    print("Inizializzazione del training setup...")
    
    # 1. Caricamento e Split del Dataset
    dataset = LongitudinalDataset("static_features.csv", "dynamic_features.csv")
    
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size
    
    # Fissiamo il seed per riproducibilità
    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)
    
    print(f"\n--- DATASET SPLIT ---")
    print(f"Pazienti totali: {total_size}")
    print(f"Training Set (80%): {train_size} pazienti")
    print(f"Test Set (20%): {test_size} pazienti")
    
    # Aumentiamo la batch_size a 64 
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    
    # 2. Inizializzazione Modello
    model = MultimodalDiseasePredictor()
    
    # 3. Bilanciamento Classi (calcolato globalmente)
    num_negatives = (dataset.static_df['LABEL'] == 0).sum()
    num_positives = (dataset.static_df['LABEL'] == 1).sum()
    pos_weight = torch.tensor([num_negatives / num_positives], dtype=torch.float32)
    print(f"\nPeso applicato alla classe positiva (Diabetici): {pos_weight.item():.2f}")
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # 4. Loop di Addestramento e Validazione
    epochs = 15
    print("\nInizio addestramento formale...")
    
    for epoch in range(epochs):
        # --- FASE DI TRAINING ---
        model.train()
        train_loss = 0.0
        
        for static_b, dyn_b, mask_b, labels_b in train_loader:
            optimizer.zero_grad()
            logits = model(static_b, dyn_b, mask_b).squeeze()
            if logits.dim() == 0:
                logits, labels_b = logits.unsqueeze(0), labels_b.unsqueeze(0)
            
            loss = criterion(logits, labels_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # --- FASE DI TEST (EVALUATION) ---
        model.eval()
        test_loss = 0.0
        test_labels = []
        test_preds = []
        
        # torch.no_grad() spegne il calcolo delle derivate: risparmia memoria ed evita il data leakage
        with torch.no_grad():
            for static_b, dyn_b, mask_b, labels_b in test_loader:
                logits = model(static_b, dyn_b, mask_b).squeeze()
                if logits.dim() == 0:
                    logits, labels_b = logits.unsqueeze(0), labels_b.unsqueeze(0)
                
                loss = criterion(logits, labels_b)
                test_loss += loss.item()
                
                probs = torch.sigmoid(logits)
                test_labels.extend(labels_b.numpy())
                test_preds.extend(probs.numpy())
                
        avg_test_loss = test_loss / len(test_loader)
        
        try:
            test_auc = roc_auc_score(test_labels, test_preds)
        except ValueError:
            test_auc = 0.5
            
        print(f"Epoca {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | Test ROC-AUC: {test_auc:.4f}")

if __name__ == "__main__":
    train_and_evaluate()
