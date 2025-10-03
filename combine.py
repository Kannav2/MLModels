import pandas as pd

files = [
    'data/data_xgboost_BGIS.csv',
    'data/data_xgboost_Carters.csv',
    'data/data_xgboost_CognirSeniorLiving.csv',
    'data/data_xgboost_CountryOfSimcoe.csv',
    'data/data_xgboost_Extendicare.csv',
    'data/data_xgboost_Johnson_Electric.csv',
    'data/data_xgboost_Kuehne_Nagel.csv',
    'data/data_xgboost_LifeLabs.csv',
    'data/data_xgboost_Rogers.csv',
    'data/data_xgboost_TJX.csv',
]

dfs = []
for file in files:
    df = pd.read_csv(file)
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)
combined.to_csv('data/combined_dataset.csv', index=False)

print(f"Combined dataset: {combined.shape[0]} rows × {combined.shape[1]} columns")
print("Saved to: data/combined_dataset.csv")
