import pandas as pd

# Read the datasets
carters = pd.read_csv('data/data_xgboost_Carters.csv')
rogers = pd.read_csv('data/data_xgboost_Rogers.csv')

print("Dataset Comparison: Carters vs Rogers")
print("=" * 50)
print(f"Carters: {carters.shape[0]} rows, {carters.shape[1]} columns")
print(f"Rogers:  {rogers.shape[0]} rows, {rogers.shape[1]} columns")
print()

# Get column differences
carters_cols = set(carters.columns)
rogers_cols = set(rogers.columns)

carters_only = carters_cols - rogers_cols
rogers_only = rogers_cols - carters_cols
common_cols = carters_cols & rogers_cols

print("Column Analysis:")
print(f"• Common columns: {len(common_cols)}")
print(f"• Only in Carters: {len(carters_only)}")
print(f"• Only in Rogers: {len(rogers_only)}")
print()

print("Columns ONLY in Carters:")
print("-" * 30)
for col in sorted(carters_only):
    print(f"• {col}")

print()
print("Columns ONLY in Rogers:")
print("-" * 30)
for col in sorted(rogers_only):
    print(f"• {col}")

print()
print("Common columns (first 10):")
print("-" * 30)
for col in sorted(common_cols)[:10]:
    print(f"• {col}")
if len(common_cols) > 10:
    print(f"... and {len(common_cols) - 10} more")
