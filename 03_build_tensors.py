import pandas as pd
import numpy as np

print("Costruzione delle matrici in corso...")
# Caricamento dataset
patients = pd.read_csv("patients.csv")
conditions = pd.read_csv("conditions.csv")
observations = pd.read_csv("observations.csv", low_memory=False)

# --- 1. COSTRUZIONE COORTE E MATRICE STATICA ---
DIABETES_CODE = 44054006 
diabetes_cond = conditions[conditions['CODE'] == DIABETES_CODE].copy()
diabetes_cond['START'] = pd.to_datetime(diabetes_cond['START']).dt.tz_localize(None)

first_diag = diabetes_cond.groupby('PATIENT')['START'].min().reset_index()
first_diag.rename(columns={'START': 'DIABETES_ONSET_DATE'}, inplace=True)

cohort = patients[['Id', 'BIRTHDATE', 'GENDER']].copy()
cohort = cohort.merge(first_diag, left_on='Id', right_on='PATIENT', how='left')

# Creazione Label (0 = Sano, 1 = Diabetico)
cohort['LABEL'] = cohort['DIABETES_ONSET_DATE'].notna().astype(int)

# Binarizzazione del sesso (M=0, F=1)
cohort['GENDER_NUM'] = (cohort['GENDER'] == 'F').astype(int)

# Calcolo dell'età
cohort['BIRTHDATE'] = pd.to_datetime(cohort['BIRTHDATE']).dt.tz_localize(None)
reference_date = cohort['DIABETES_ONSET_DATE'].fillna(pd.to_datetime('2026-08-17'))
cohort['AGE_AT_REF'] = (reference_date - cohort['BIRTHDATE']).dt.days / 365.25

static_features = cohort[['Id', 'GENDER_NUM', 'AGE_AT_REF', 'LABEL']].copy()

# --- 2. COSTRUZIONE MATRICE DINAMICA (SERIE TEMPORALI) ---
observations['DATE'] = pd.to_datetime(observations['DATE']).dt.tz_localize(None)

# Mappatura dei codici LOINC
TARGET_LOINC = {
    '4548-4': 'HbA1c', 
    '2339-0': 'Glucose', 
    '39156-5': 'BMI',
    '8480-6': 'Systolic_BP',
    '8462-4': 'Diastolic_BP'
}
obs_filtered = observations[observations['CODE'].isin(TARGET_LOINC.keys())].copy()
obs_filtered['CODE'] = obs_filtered['CODE'].map(TARGET_LOINC)

obs_merged = obs_filtered.merge(cohort[['Id', 'DIABETES_ONSET_DATE']], left_on='PATIENT', right_on='Id', how='inner')
mask_healthy = obs_merged['DIABETES_ONSET_DATE'].isna()
mask_before_onset = obs_merged['DATE'] < obs_merged['DIABETES_ONSET_DATE']
valid_obs = obs_merged[mask_healthy | mask_before_onset].copy()


valid_obs['VALUE'] = pd.to_numeric(valid_obs['VALUE'], errors='coerce')


valid_obs = valid_obs.dropna(subset=['VALUE'])

valid_obs['YEAR'] = valid_obs['DATE'].dt.year

# PIVOT TABLE
dynamic_features = valid_obs.pivot_table(
    index=['PATIENT', 'YEAR'], 
    columns='CODE', 
    values='VALUE', 
    aggfunc='mean'
).reset_index()

# Stampa dei risultati
print("\n--- MATRICE STATICA (Condizioni Iniziali) ---")
print(f"Dimensioni: {static_features.shape}")
print(static_features.head())

print("\n--- MATRICE DINAMICA (Stato per Singolo Anno) ---")
print(f"Dimensioni: {dynamic_features.shape}")
print(dynamic_features.head(10))
static_features.to_csv("static_features.csv", index=False)
dynamic_features.to_csv("dynamic_features.csv", index=False)
print("Matrici salvate con successo in CSV!")
