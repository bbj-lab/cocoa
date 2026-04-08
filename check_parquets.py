import polars as pl

df_meds = pl.read_parquet("data/processed/meds.parquet")
print("\nmeds.parquet — first 5 rows:")
print(df_meds.head())  # Views the first few rows in the terminal

df_subject_splits = pl.read_parquet("data/processed/subject_splits.parquet")
print("\nsubject_splits.parquet — first 5 rows:")
print(df_subject_splits.head())  # Views the first few rows in the terminal

df_tokens_times = pl.read_parquet("data/processed/tokens_times.parquet")
print("\ntokens_times.parquet — first 5 rows:")
print(df_tokens_times.head())  # Views the first few rows in the terminal

# LABEL PARQUET CHECKS
df_pressor_escalation_events = pl.read_parquet("data/processed/pressor_escalation_events.parquet")
print("\npressor_escalation_events.parquet — first 5 rows:")
print(df_pressor_escalation_events.head())  # Views the first few rows in the terminal

df_hosp_mortality_events = pl.read_parquet("data/processed/hospital_mortality_events.parquet")
print("\df_hosp_mortality_events.parquet — first 5 rows:")
print(df_hosp_mortality_events.head())  # Views the first few rows in the terminal