import torch
import torch.nn as nn

class MultimodalDiseasePredictor(nn.Module):
    def __init__(self, static_input_dim=2, dyn_input_dim=5, lstm_hidden_dim=32, static_hidden_dim=16):
        super(MultimodalDiseasePredictor, self).__init__()
        
        # 1. Ramo Statico (MLP per età e genere)
        self.static_encoder = nn.Sequential(
            nn.Linear(static_input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, static_hidden_dim),
            nn.ReLU()
        )
        
        # 2. Ramo Dinamico (LSTM per le serie temporali cliniche)
        # batch_first=True indica che i tensori hanno dimensione [batch, timesteps, features]
        self.lstm = nn.LSTM(
            input_size=dyn_input_dim, 
            hidden_size=lstm_hidden_dim, 
            num_layers=1, 
            batch_first=True
        )
        
        # 3. Classificatore Finale (Late Fusion)
        # Prende in input la concatenazione dei due rami: 16 (statico) + 32 (dinamico) = 48
        fusion_dim = static_hidden_dim + lstm_hidden_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2), # Previene l'overfitting forzando la rete a non dipendere troppo da un solo parametro
            nn.Linear(32, 1) # Singolo output: valore non normalizzato (Logit) per la Binary Cross Entropy
        )

    def forward(self, static_x, dyn_x, mask):
        # --- Elaborazione Dati Statici ---
        static_out = self.static_encoder(static_x) # [batch_size, static_hidden_dim]
        
        # --- Elaborazione Serie Temporali ---
        lstm_out, _ = self.lstm(dyn_x) # lstm_out shape: [batch_size, max_len, lstm_hidden_dim]
        
        # RECUPERO DELL'ULTIMO STATO VALIDO:
        # Usiamo la 'mask' per contare quante misurazioni reali ha fatto il paziente
        batch_size = dyn_x.size(0)
        lengths = mask.sum(dim=1).long() 
        
        # Sottraiamo 1 per ottenere l'indice dell'ultimo esame (clamp(min=0) evita errori per chi ha 0 esami)
        last_valid_indices = (lengths - 1).clamp(min=0) 
        
        # Estraiamo lo stato vettoriale corrispondente SOLO all'ultimo esame reale
        dyn_out = lstm_out[torch.arange(batch_size), last_valid_indices, :] # [batch_size, lstm_hidden_dim]
        
        # --- FUSIONE E PREDIZIONE ---
        # Concateniamo i due vettori lungo l'asse delle feature
        combined = torch.cat((static_out, dyn_out), dim=1) # [batch_size, fusion_dim]
        
        # Calcoliamo la predizione finale
        logits = self.classifier(combined)
        
        return logits

# --- TEST DELL'ARCHITETTURA ---
if __name__ == "__main__":
    print("Test del flusso tensoriale in corso...")
    
    # Simuliamo un batch in ingresso con le stesse dimensioni fornite dal Dataloader
    batch_size = 16
    max_len = 10
    
    dummy_static = torch.randn(batch_size, 2)
    dummy_dyn = torch.randn(batch_size, max_len, 5)
    
    # Simuliamo una maschera temporale (es. il primo paziente ha 4 esami validi, il secondo 7, ecc.)
    dummy_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    dummy_mask[0, :4] = True
    dummy_mask[1, :7] = True
    dummy_mask[2:, :2] = True # Gli altri ne hanno solo 2
    
    # Inizializziamo il modello
    model = MultimodalDiseasePredictor()
    
    # Forward pass
    output = model(dummy_static, dummy_dyn, dummy_mask)
    
    print("\n--- RISULTATO DEL FORWARD PASS ---")
    print(f"Dimensione dell'output finale: {output.shape}")
    print("Se la dimensione è [16, 1], la logica di concatenamento e l'estrazione temporale funzionano perfettamente.")
