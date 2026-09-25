import pandas as pd

# Caricamento dati
print("Caricamento dataset in corso...")
patients = pd.read_csv("patients.csv")
conditions = pd.read_csv("conditions.csv")

# Codice SNOMED per il Diabete T2 in Synthea
DIABETES_CODE = 44054006 

# Filtriamo le diagnosi e prendiamo la prima insorgenza
diabetes_cond = conditions[conditions['CODE'] == DIABETES_CODE].copy()
diabetes_cond['START'] = pd.to_datetime(diabetes_cond['START'])
first_diagnosis = diabetes_cond.groupby('PATIENT')['START'].min().reset_index()
first_diagnosis.rename(columns={'START': 'DIABETES_ONSET_DATE'}, inplace=True)

# Uniamo i dati anagrafici con le diagnosi
cohort = patients[['Id', 'BIRTHDATE', 'GENDER']].copy()
cohort = cohort.merge(first_diagnosis, left_on='Id', right_on='PATIENT', how='left')

# Creazione della Label: 1 per i diabetici, 0 per i sani
cohort['LABEL'] = cohort['DIABETES_ONSET_DATE'].notna().astype(int)
cohort.drop(columns=['PATIENT'], inplace=True)

# Output dei risultati
print("\n--- STATISTICHE COORTE ---")
print(f"Pazienti totali estratti: {len(cohort)}")
print(f"Di cui diabetici (Classe 1): {cohort['LABEL'].sum()}")
print(f"Di cui sani (Classe 0): {(cohort['LABEL'] == 0).sum()}")