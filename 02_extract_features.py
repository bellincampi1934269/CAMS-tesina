import pandas as pd

print("Caricamento dataset in corso (potrebbe richiedere qualche secondo)...")
patients = pd.read_csv("patients.csv")
conditions = pd.read_csv("conditions.csv")
observations = pd.read_csv("observations.csv")

# 1. Ricreiamo la coorte
DIABETES_CODE = 44054006 
diabetes_cond = conditions[conditions['CODE'] == DIABETES_CODE].copy()
diabetes_cond['START'] = pd.to_datetime(diabetes_cond['START']).dt.tz_localize(None)
first_diagnosis = diabetes_cond.groupby('PATIENT')['START'].min().reset_index()
first_diagnosis.rename(columns={'START': 'DIABETES_ONSET_DATE'}, inplace=True)

cohort = patients[['Id', 'BIRTHDATE', 'GENDER']].copy()
cohort = cohort.merge(first_diagnosis, left_on='Id', right_on='PATIENT', how='left')

# 2. Prepariamo le osservazioni 
observations['DATE'] = pd.to_datetime(observations['DATE']).dt.tz_localize(None)

# Codici LOINC per esami chiave del Diabete
TARGET_LOINC = ['4548-4', '2339-0', '39156-5']
obs_filtered = observations[observations['CODE'].isin(TARGET_LOINC)].copy()

# 3. Uniamo le osservazioni alla coorte
obs_merged = obs_filtered.merge(cohort[['Id', 'DIABETES_ONSET_DATE']], left_on='PATIENT', right_on='Id', how='inner')

# 4. PREVENZIONE DATA LEAKAGE: 
mask_healthy = obs_merged['DIABETES_ONSET_DATE'].isna()
mask_before_onset = obs_merged['DATE'] < obs_merged['DIABETES_ONSET_DATE']

final_observations = obs_merged[mask_healthy | mask_before_onset].copy()

# Statistiche finali
print("\n--- STATISTICHE SERIE TEMPORALI ---")
print(f"Misurazioni totali estratte (HbA1c, Glucosio, BMI): {len(final_observations)}")

diabetic_obs = final_observations[final_observations['DIABETES_ONSET_DATE'].notna()]
avg_obs_diabetic = diabetic_obs.groupby('PATIENT').size().mean()

print(f"Media di misurazioni disponibili per paziente diabetico (prima della diagnosi): {avg_obs_diabetic:.1f}")
print("\nPrime righe delle misurazioni pulite:")
print(final_observations[['PATIENT', 'DATE', 'DESCRIPTION', 'VALUE', 'UNITS']].head())
