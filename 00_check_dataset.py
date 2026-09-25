import pandas as pd

print("Controllo del dataset esteso in corso (potrebbe richiedere qualche secondo)...\n")

# 1. Controllo pazienti
patients = pd.read_csv("patients.csv")
print(f"--- PAZIENTI ---")
print(f"Totale pazienti nel file: {len(patients)}")

# 2. Controllo Diabete
conditions = pd.read_csv("conditions.csv")
# Il codice SNOMED del diabete usato
diabetes_cond = conditions[conditions['CODE'] == 44054006]
print(f"\n--- CONDIZIONI CLINICHE ---")
print(f"Totale diagnosi di Diabete trovate: {len(diabetes_cond)}")
print(f"Prevalenza stimata: {(len(diabetes_cond)/len(patients))*100:.2f}%")

# 3. Controllo Esami Clinici (Observations)
observations = pd.read_csv("observations.csv", low_memory=False)
TARGET_LOINC = {
    '4548-4': 'HbA1c (Emoglobina Glicata)', 
    '2339-0': 'Glucosio', 
    '39156-5': 'BMI (Indice Massa Corporea)',
    '8480-6': 'Pressione Sistolica',
    '8462-4': 'Pressione Diastolica'
}

obs_filtered = observations[observations['CODE'].isin(TARGET_LOINC.keys())].copy()
print(f"\n--- ESAMI CLINICI (OBSERVATIONS) ---")
print(f"Totale esami vitali utili trovati: {len(obs_filtered)}")

print("\nDettaglio per tipo di esame:")
# Mappiamo i codici
counts = obs_filtered['CODE'].map(TARGET_LOINC).value_counts()
print(counts)
