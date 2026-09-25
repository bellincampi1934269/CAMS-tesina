import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class LongitudinalDataset(Dataset):
    def __init__(self, static_csv, dynamic_csv):
        print("Caricamento dati in memoria...")
        self.static_df = pd.read_csv(static_csv)
        dynamic_df = pd.read_csv(dynamic_csv)
        
        # 1. IMPUTAZIONE (Zero-Order Hold / Forward Fill)
        print("Applicazione del Forward Fill (Zero-Order Hold) sui dati mancanti...")
        
        # Identifichiamo le colonne cliniche da imputare
        self.dyn_cols = [c for c in dynamic_df.columns if c not in ['PATIENT', 'YEAR']]
        
        # Applichiamo il ffill e bfill solo alle colonne degli esami, raggruppando per paziente
        dynamic_df[self.dyn_cols] = dynamic_df.groupby('PATIENT')[self.dyn_cols].ffill().bfill()
        
        # Se un esame non è MAI stato fatto da un paziente, riempiamo con 0
        dynamic_df[self.dyn_cols] = dynamic_df[self.dyn_cols].fillna(0)
        
        self.patients = self.static_df['Id'].unique()
        self.static_data = self.static_df.set_index('Id')
        self.dynamic_data = dynamic_df

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patient_id = self.patients[idx]
        
        # --- ESTRAZIONE DATI STATICI ---
        static_row = self.static_data.loc[patient_id]
        static_tensor = torch.tensor([static_row['GENDER_NUM'], static_row['AGE_AT_REF']], dtype=torch.float32)
        label = torch.tensor(static_row['LABEL'], dtype=torch.float32)

        # --- ESTRAZIONE SERIE TEMPORALE ---
        patient_dyn = self.dynamic_data[self.dynamic_data['PATIENT'] == patient_id]
        
        if len(patient_dyn) == 0:
            dyn_tensor = torch.zeros((1, len(self.dyn_cols)), dtype=torch.float32)
        else:
            patient_dyn = patient_dyn.sort_values('YEAR')
            dyn_tensor = torch.tensor(patient_dyn[self.dyn_cols].values, dtype=torch.float32)

        return static_tensor, dyn_tensor, label

# 2. PADDING E BATCHING
def collate_fn(batch):
    static_list, dyn_list, labels_list = zip(*batch)
    
    static_batch = torch.stack(static_list)
    labels_batch = torch.stack(labels_list)
    
    # Padding: allineiamo le matrici aggiungendo zeri
    dyn_batch = pad_sequence(dyn_list, batch_first=True, padding_value=0.0)
    
    # Creazione della Mask
    lengths = torch.tensor([len(seq) for seq in dyn_list])
    batch_size, max_len, _ = dyn_batch.shape
    mask = torch.arange(max_len).expand(batch_size, max_len) < lengths.unsqueeze(1)
    
    return static_batch, dyn_batch, mask, labels_batch

# --- TEST DEL DATALOADER ---
if __name__ == "__main__":
    dataset = LongitudinalDataset("static_features.csv", "dynamic_features.csv")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    
    static_b, dyn_b, mask_b, labels_b = next(iter(dataloader))
    
    print("\n--- TEST DATALOADER PYTORCH ---")
    print(f"Batch Dati Statici (Pazienti, Feature): {static_b.shape}")
    print(f"Batch Dati Dinamici (Pazienti, Timesteps, Feature): {dyn_b.shape}")
    print(f"Batch Mask (Pazienti, Timesteps): {mask_b.shape}")
    print(f"Batch Labels (Pazienti): {labels_b.shape}")
