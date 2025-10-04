# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import toml

config = toml.load("/Users/kannav/Desktop/tmp/AcclaimProjects/DatasetAnalysisEDA/config.toml")

# %%
file_path = config["data_paths"]["first_csv_file"]
df = pd.read_csv("/Users/kannav/Desktop/tmp/AcclaimProjects/DatasetAnalysisEDA/data/combined_dataset.csv")

# %%
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# %%
df.info(verbose=True,show_counts=True)

# %%
# Categories: Empty columns (0 non-null), Very low coverage (<5%), Identifiers, Too granular

# empty columns
empty_columns = [
    'Parent Company',
    'Medical Update Due Date', 
    'Offer Of Modified/Condition',
    'Date Closed',
    'Group 1 (Cost Centre/Class/Division)',
    'Group 2 (Org Code/Region)', 
    'Group 3 (Location Code/Store #)',
    'Group 4 (Department/Banner)',
    'Recommended RTW-Reg',
    'Broker',
    'SAM', 
    'LOA Code',
    'Next Action Due Date',
    'COVID Impact Type of Leave',
    'Legislative Leave Type of Leave', 
    'Benefit End Date',
    'Reason for Contacting EAP',
    'Director',
    'Send LTD Transfer Notice Date',
    'Claim Number',
    'Entitlement Status', 
    'Rogers Incident Number',
    'Medical Aid Type',
    'QA Name'
]

# Very low coverage columns (<5% non-null)
low_coverage_columns = [
    'Date of Injury/Illness',  # 26/6038 = 0.4% - maybe not to drop
    'Rate',  # 44/6038 = 0.7% 
    'EI Benefit Start Date',  # 1/6038 = 0.02%
    'Union',  # 22/6038 = 0.4%
    'Supervisor Name'  # 21/6038 = 0.3%
]

# Identifier columns (not useful for prediction)
identifier_columns = [
    'Employee Name', # useful, don't drop it , anonymise it, eventually
    'Employee ID'
]

# Too granular/administrative columns
administrative_columns = [
    'Manager Name',  
    'People Leader',  
    'Primary Consultant',  
    'HR Name'  
]

period_columns_to_drop = [
    'Modified Days this Period',
    'Suspended Days this Period', 
    'Calendar Days Lost Days this Period (Modified Days Removed)',
    'Calendar Days Lost Days this Period',
    'Lost Days this Period',
    'Lost Days this Period (Modified Days Removed)',
    'Managed Value',
    'Position'
]

columns_to_drop = empty_columns + low_coverage_columns + identifier_columns + administrative_columns + period_columns_to_drop

print(f"Total columns to drop: {len(columns_to_drop)}")
print(f"Original dataset shape: {df.shape}")
print("\nColumns being dropped:")
for i, col in enumerate(columns_to_drop, 1):
    print(f"{i:2d}. {col}")


# %%
df.shape

# %%
df_cleaned = df.drop(columns=columns_to_drop, errors='ignore')

print(f"Original shape: {df.shape}")
print(f"Cleaned shape: {df_cleaned.shape}")
print(f"Columns reduced by: {df.shape[1] - df_cleaned.shape[1]}")


print(f"\nRemaining {df_cleaned.shape[1]} columns:")
remaining_columns = list(df_cleaned.columns)
for i, col in enumerate(remaining_columns, 1):
    print(f"{i:2d}. {col}")


# %%
print(df_cleaned["Claim Decision"].value_counts())
print(df_cleaned["Claim Decision"].value_counts().sum())

# %% [markdown]
# so 7436 is the number of rows with not null values in Claim Decision

# %%
df_cleaned.info(verbose=True)

# %%
from matplotlib import pyplot as plt

# Create pie chart with percentages
plt.figure(figsize=(8, 6))
plt.pie(df_cleaned.dtypes.value_counts(), 
        labels=df_cleaned.dtypes.value_counts().index,
        autopct='%1.1f%%',  
        startangle=90)      
plt.title("Data Types in DataFrame")
plt.axis('equal') 
plt.show()


# %%
df_cleaned.dropna(subset=["Claim Decision"], inplace=True)

print(df_cleaned["Claim Decision"].value_counts())
print(df_cleaned["Claim Decision"].value_counts().sum())



# %% [markdown]
# drop this suspended claim maybe?

# %%
df_cleaned.shape

# %%
df_cleaned["Claim Decision"].value_counts().plot(kind='bar',figsize=(10,5))


plt.legend(title='Claim Decision', labels=['Claimed', 'Not Claimed'])
plt.title('Claim Decision Distribution')
plt.xlabel('Claim Decision')
plt.ylabel('Count')
plt.show()

# %%
numerical_columns = df_cleaned.select_dtypes(include=['number']).columns
for col in numerical_columns:
    print(col)

# %%
correlation_matrix = df_cleaned[numerical_columns].corr()

plt.figure(figsize=(15, 12))

sns.heatmap(correlation_matrix, 
            annot=True,           
            cmap='coolwarm',      
            center=0,             
            square=True,          
            fmt='.2f',            
            cbar_kws={"shrink": .8})  

plt.title('Correlation Matrix Heatmap - Numerical Variables', 
          fontsize=16, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right')  
plt.yticks(rotation=0)               
plt.tight_layout()                   
plt.show()

print(f"Correlation matrix shape: {correlation_matrix.shape}")

# %%
data_columns = ["First Day Absent","Benefit Start Date","Date Medical Received","Date Received by Acclaim","Current Date Claim Approved Until ","RTW Modified Start Date","RTW Modified End Date","RTW Modified Formula Date","Suspended Benefit Date","Reinstated Benefit Date","Suspended Benefit Formula Date","Actual RTW Regular Date","End Date","Formula Date","LTD Start Date","Date of Birth","Date of Hire","Suggested RTW-Mod"]

for col in data_columns:
    df_cleaned[col] = pd.to_datetime(df_cleaned[col],errors="coerce")

len(df_cleaned.select_dtypes(include=['datetime']).columns)


# %%
import prince

injury_columns = ['Injury/Illness Category', 'Nature of Injury/Illness', 'Body Part']
mca = prince.MCA(n_components=2,n_iter=3,random_state=42,engine="sklearn")

mca_fitted = mca.fit(df_cleaned[injury_columns])
print(mca_fitted.eigenvalues_summary)

mca_components = mca_fitted.transform(df_cleaned[injury_columns])

print(f"MCA components type: {type(mca_components)}")
print(f"MCA components shape: {mca_components.shape}")
print(f"MCA components columns: {mca_components.columns.tolist()}")

# Extract first component (pandas DataFrame indexing)
first_component = mca_components.iloc[:, 0]
second_component = mca_components.iloc[:, 1]

print(f"First component shape: {first_component.shape}")
print(f"First component type: {type(first_component)}")








# %%
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
claim_decision_encoded = le.fit_transform(df_cleaned['Claim Decision'])

corr_df = pd.DataFrame({
    'first_component': first_component,
    'second_component': second_component,
    'claim_decision_encoded': claim_decision_encoded
})


corr_data = corr_df.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, 
            square=True, fmt='.3f', cbar_kws={"shrink": .8})
plt.title('Correlation: First Component vs Claim Decision')
plt.tight_layout()
plt.show()

print("Encoding mapping:")
for i, label in enumerate(le.classes_):
    print(f"{i}: {label}")


# %%

print("NUMERICAL COLUMNS:")
print(df_cleaned.select_dtypes(include=['number']).columns.tolist())

print("\nCATEGORICAL/OBJECT COLUMNS:")  
print(df_cleaned.select_dtypes(include=['object']).columns.tolist())

print("\nDATETIME COLUMNS:")
print(df_cleaned.select_dtypes(include=['datetime']).columns.tolist())



# %%
print("Before transformations:")
print(f"Lapse Days dtype: {df_cleaned['Lapse Days'].dtype}")
print(f"Cost column dtype: {df_cleaned['Cost of Total Lost Days (Modified Days Removed)'].dtype}")
print(f"Dataset shape: {df_cleaned.shape}")

df_cleaned['Lapse Days'] = pd.to_numeric(df_cleaned['Lapse Days'], errors='coerce')

df_cleaned['Cost of Total Lost Days (Modified Days Removed)'] = (
    df_cleaned['Cost of Total Lost Days (Modified Days Removed)']
    .astype(str)
    .str.replace('$', '', regex=False)
    .str.replace(',', '', regex=False)
    .str.strip()
)
df_cleaned['Cost of Total Lost Days (Modified Days Removed)'] = pd.to_numeric(
    df_cleaned['Cost of Total Lost Days (Modified Days Removed)'], errors='coerce'
)

# 3. Remove New and Closed columns
df_cleaned = df_cleaned.drop(columns=['New', 'Closed'])

print("\nAfter transformations:")
print(f"Lapse Days dtype: {df_cleaned['Lapse Days'].dtype}")
print(f"Cost column dtype: {df_cleaned['Cost of Total Lost Days (Modified Days Removed)'].dtype}")
print(f"Dataset shape: {df_cleaned.shape}")
print(f"Columns removed: New, Closed")



# %% [markdown]
# Injury_Nature_BodyPart

# %% [markdown]
# investigate body part for unspecified, body part - myabe consultants?? , find unspecified body part numbers and percentages and numerical total

# %%
df_cleaned["Nature of Injury/Illness"].value_counts()
df_cleaned["Nature of Injury/Illness"].fillna("Other",inplace=True)
df_cleaned["Nature of Injury/Illness"].value_counts()

# %%
df_cleaned["combined_injury"] = df_cleaned["Injury/Illness Category"] + "_" + df_cleaned["Nature of Injury/Illness"] + "_" + df_cleaned["Body Part"]

combinations = df_cleaned["combined_injury"].value_counts()


# %%
# Visualize cutoff for "Other" category - minimal code
counts = df_cleaned["combined_injury"].value_counts()

# Plot frequency distribution
plt.figure(figsize=(12, 6))
plt.plot(range(len(counts)), counts.values, 'b-', linewidth=2)
plt.axhline(y=50, color='r', linestyle='--', label='50 threshold')
plt.axhline(y=100, color='orange', linestyle='--', label='100 threshold')
plt.xlabel('Category Rank')
plt.ylabel('Frequency')
plt.title('Category Frequency Distribution - Cutoff Visualization')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Show cutoff statistics
print(f"Total categories: {len(counts)}")
print(f"Categories with >100 occurrences: {(counts > 100).sum()}")
print(f"Categories with >50 occurrences: {(counts > 50).sum()}")
print(f"Categories with >20 occurrences: {(counts > 20).sum()}")
print(f"Cumulative % with >20 occurrences: {(counts[counts > 20].sum() / counts.sum() * 100):.1f}%")


# %%
counts = df_cleaned["combined_injury"].value_counts()
low_count_categories = counts[counts < 20].index


df_cleaned["combined_injury"] = df_cleaned["combined_injury"].replace(low_count_categories, "Other")

print(f"Categories changed to 'Other': {len(low_count_categories)}")
print(f"New unique categories: {df_cleaned['combined_injury'].nunique()}")
print(f"'Other' category count: {(df_cleaned['combined_injury'] == 'Other').sum()}")
print("\nNew value counts:")
print(df_cleaned["combined_injury"].value_counts().head(10))


# %%
from category_encoders import BinaryEncoder

# Apply binary encoding to combined_injury using sklearn
binary_encoder = BinaryEncoder()
binary_encoded_injury = binary_encoder.fit_transform(df_cleaned['combined_injury'])

print(f"Original combined_injury has {df_cleaned['combined_injury'].nunique()} unique categories")
print(f"Binary encoding created {binary_encoded_injury.shape[1]} binary columns")
print(f"Binary encoded shape: {binary_encoded_injury.shape}")
print("\nBinary encoded columns:")
print(binary_encoded_injury.columns.tolist())
print("\nFirst few rows of binary encoding:")
print(binary_encoded_injury.head())


# %%
from sklearn.feature_selection import mutual_info_classif
from category_encoders import BinaryEncoder
from sklearn.preprocessing import LabelEncoder

# Quick setup of encodings
binary_encoder = BinaryEncoder()
binary_encoded = binary_encoder.fit_transform(df_cleaned['combined_injury'])

label_encoder = LabelEncoder()
label_encoded = label_encoder.fit_transform(df_cleaned['combined_injury'])

claim_decision_encoded = LabelEncoder().fit_transform(df_cleaned['Claim Decision'])

# Calculate mutual information
mi_binary = mutual_info_classif(binary_encoded, claim_decision_encoded, random_state=42)
mi_label = mutual_info_classif(label_encoded.reshape(-1, 1), claim_decision_encoded, random_state=42)

# Results
print("MUTUAL INFORMATION RESULTS:")
print("="*40)
print(f"Binary Encoding MI (per column):")
for i, mi_val in enumerate(mi_binary):
    print(f"  {binary_encoded.columns[i]}: {mi_val:.4f}")
print(f"Binary Encoding MI (total): {mi_binary.sum():.4f}")
print(f"Label Encoding MI: {mi_label[0]:.4f}")
print("="*40)


# %%
df_cleaned['secondary_diagnosis_binary'] = df_cleaned['Secondary Diagnosis'].notna().astype(int)

print("Secondary Diagnosis Binary Conversion:")
print(f"Has secondary diagnosis (1): {df_cleaned['secondary_diagnosis_binary'].sum()}")
print(f"No secondary diagnosis (0): {(df_cleaned['secondary_diagnosis_binary'] == 0).sum()}")
print(f"Percentage with secondary diagnosis: {df_cleaned['secondary_diagnosis_binary'].mean()*100:.1f}%")

print("\nSample of original vs binary:")
sample = df_cleaned[['Secondary Diagnosis', 'secondary_diagnosis_binary']].head(10)


# %%
from sklearn.feature_selection import mutual_info_classif

mi_secondary = mutual_info_classif(
    df_cleaned[['secondary_diagnosis_binary']], 
    claim_decision_encoded, 
    random_state=42
)

combined_features = pd.DataFrame({
    'secondary_diagnosis_binary': df_cleaned['secondary_diagnosis_binary'],
    'combined_injury_label': label_encoded
})

mi_combined = mutual_info_classif(combined_features, claim_decision_encoded, random_state=42)

print("MUTUAL INFORMATION RESULTS:")
print("="*50)
print(f"Secondary Diagnosis alone: {mi_secondary[0]:.4f}")
print(f"Combined Injury alone: {mi_label[0]:.4f}")
print(f"Secondary + Injury combined: {mi_combined.sum():.4f}")
print(f"Improvement: {((mi_combined.sum() - mi_label[0])/mi_label[0]*100):+.1f}%")
print("\nFeature breakdown:")
print(f"  Secondary diagnosis: {mi_combined[0]:.4f}")
print(f"  Combined injury: {mi_combined[1]:.4f}")
print("="*50)


# %%
print(df_cleaned.info(verbose=True,show_counts=True))

# %%
df_cleaned['first_day_absent_missing'] = df_cleaned['First Day Absent'].isna().astype(int)
df_cleaned['benefit_start_missing'] = df_cleaned['Benefit Start Date'].isna().astype(int)
df_cleaned['medical_received_missing'] = df_cleaned['Date Medical Received'].isna().astype(int)

# Drop the original date columns (too many missing values)
df_cleaned = df_cleaned.drop(columns=[
    'First Day Absent', 
    'Benefit Start Date', 
    'Date Medical Received'
])

print("Replaced 3 date columns with 3 binary indicators")
print(f"Dataset shape: {df_cleaned.shape}")

# %%
print(df_cleaned["Surgical Intervention"].value_counts())

df_cleaned["Surgical Intervention"] = (df_cleaned["Surgical Intervention"] == "Yes").astype(int)

print(df_cleaned["Surgical Intervention"].value_counts())

# %%
print(df_cleaned["Body Part"].value_counts())
# leave it alone
print(df_cleaned["Mechanism of Injury"].value_counts())

# unspecified maybe mental health issues, check and fix it - anxiety, ptsd, depression, no physical injury as well

# %% [markdown]
# flag this 

# %%
df_cleaned["Body Part"] = (df_cleaned["Body Part"] != "Unspecified").astype(int)
print(df_cleaned["Body Part"].value_counts())

# %%
df_cleaned["medical_complexity"] = (
    df_cleaned["secondary_diagnosis_binary"] + 
    df_cleaned["Surgical Intervention"] + 
    df_cleaned["Body Part"]
)

print(df_cleaned["medical_complexity"].value_counts())

# %%
print(df_cleaned["New Value"].value_counts())
print(df_cleaned["Closed Value"].value_counts())

# %%
print(df_cleaned["Claim History"].value_counts())

# %%
df_cleaned.info(verbose=True,show_counts=True)

# %%
# RTW Modified Start Date                                1192 non-null   datetime64[ns]
#  8   RTW Modified End Date 
print(df_cleaned["RTW Modified Start Date"].value_counts())
print(df_cleaned["RTW Modified End Date"].value_counts())
print(df_cleaned["RTW Modified Formula Date"].value_counts())
print(df_cleaned["Suspended Benefit Date"].value_counts())
print(df_cleaned["Reinstated Benefit Date"].value_counts())
print(df_cleaned["Suspended Benefit Formula Date"].value_counts())
print(df_cleaned["End Date"].value_counts())
# print(df_cleaned["Reason"].value_counts())
print(df_cleaned["Body Part"].value_counts())
print(df_cleaned["Generation"].value_counts())
print(df_cleaned[df_cleaned["Reason"].notna()]["Claim Decision"].value_counts())
print(df_cleaned["combined_injury"].value_counts())
print(df_cleaned["Injury/Illness Category"].value_counts())
print(df_cleaned["Benefit/Claim Type Filter"].value_counts())
# drop date of birth and date of hire as well

# we can drop this maybe?, city as well

df_cleaned = df_cleaned.drop(columns=["RTW Modified Start Date","RTW Modified End Date","RTW Modified Formula Date","Suspended Benefit Date","Reinstated Benefit Date","Suspended Benefit Formula Date","End Date","Body Part","Generation"])

# %%
df_cleaned.info(verbose=True,show_counts=True)

# %%
df_cleaned = df_cleaned.drop(columns=["Suggested RTW-Mod"])
# drop city, keep province and make it much more better, from a lot of different provinces, to less categories
df_cleaned = df_cleaned.drop(columns=["City"])

def create_geagraphic_distribution(row):
    if row["Province"] in ["Ontario","Quebec"]:
        return "Central Canada"
    elif row["Province"] in ["British Columbia"]:
        return "West Coast"
    elif row["Province"] in ["Alberta","Saskatchewan","Manitoba"]:
        return "Prairies"
    elif row["Province"] in ["New Brunswick","Nova Scotia","Prince Edward Island","Newfoundland and Labrador"]:
        return "Atlantic Canada"
    else:
        return "Other" #Unknown
        

# %%
print(df_cleaned["Province"].value_counts())
df_cleaned["Province"].replace("ON","Ontario",inplace=True)
df_cleaned["Province"].replace("BC","British Columbia",inplace=True)
print(df_cleaned["Province"].value_counts())



# %%
df_cleaned["Province"].value_counts()

# %%
df_cleaned["geographic_distribution"] = df_cleaned.apply(create_geagraphic_distribution,axis=1)
print(df_cleaned["geographic_distribution"].value_counts())

# %%
print(df_cleaned["Weeks Open"].value_counts())
print(df_cleaned["Duration in Weeks (Calendar Days)"].value_counts())

# %%
df_cleaned["has_rejected_reason"] = df["Reason"].notna().astype(int)
print(df_cleaned["has_rejected_reason"].value_counts())
df_cleaned.drop(columns=["Reason"],inplace=True)

# %%
df_cleaned.drop(columns=["Province","Injury/Illness Category Filter","Surgical Intervention","Mechanism of Injury","Claim History"],inplace=True)

# %%
df_cleaned.info(verbose=True,show_counts=True)

# %%
for col in df_cleaned.columns:
    print(col)
    print("-"*100)

# Fix: Use the correct column name with trailing space
df_cleaned["approval_time"] = (df_cleaned["Current Date Claim Approved Until "] - df_cleaned["Date Received by Acclaim"]).dt.days.clip(lower=0).fillna(0)
df_cleaned["rtw_time"] = (df_cleaned["Actual RTW Regular Date"] - df_cleaned["Date Received by Acclaim"]).dt.days.clip(lower=0).fillna(-999)

print(df_cleaned["approval_time"].value_counts())
print(df_cleaned["rtw_time"].value_counts())

print(df_cleaned["approval_time"].median())
print(df_cleaned["rtw_time"].median())





# %%
def categorize_days_to_approval(days):
    if days == 0:
        return "Not_Approved"
    elif days <= 30:
        return "Fast_Approval"
    elif days <= 90:
        return "Medium_Approval"
    else:
        return "Slow_Approval"

def categorize_days_to_rtw(days):
    if days == -999:
        return "Not_RTW"
    elif days <= 60:
        return "Quick_RTW"
    elif days <= 180:
        return "Medium_RTW"
    else:
        return "Long_RTW"

df_cleaned["approval_time_category"] = df_cleaned["approval_time"].apply(categorize_days_to_approval)
df_cleaned["rtw_time_category"] = df_cleaned["rtw_time"].apply(categorize_days_to_rtw)

print(df_cleaned["approval_time_category"].value_counts().sum())
print(df_cleaned["rtw_time_category"].value_counts().sum())



# %%
df_cleaned.info(verbose=True,show_counts=True)
# Current Date Claim Approved Until, Actual RTW Regular, LTD Start Date, Job Status, Date of birth, datre of hire
df_cleaned["has_ltd"] = df_cleaned["LTD Start Date"].notna().astype(int)

# %%
df_cleaned["Employment Status/Employment Type"].replace("F","Full-Time",inplace=True)
df_cleaned["Employment Status/Employment Type"].fillna("Unknown",inplace=True)

df_cleaned["Employment Status/Employment Type"].value_counts()

# %%
df_cleaned["Work Related"].fillna("Not Provided",inplace=True)
df_cleaned["Work Related"].value_counts()

# %%
df_cleaned.info(verbose=True,show_counts=True)

# %%
df_cleaned.drop(columns=["LTD Start Date","Actual RTW Regular Date","Secondary Diagnosis","Current Date Claim Approved Until ","Job Status/Position Status","Date of Hire","Date of Birth"],inplace=True)

# %%
df_cleaned.info(verbose=True,show_counts=True)

# %%
df_cleaned = df_cleaned[df_cleaned["Claim Decision"] != "Suspended"]
df_cleaned = df_cleaned[df_cleaned["Claim Decision"] != "COD Approved"]

print(df_cleaned["Claim Decision"].value_counts())

# %%
df_cleaned["Lapse Days"].fillna(0,inplace=True)

# %%
df_cleaned = df_cleaned.dropna()
print(df_cleaned.shape)

# %%
print(df_cleaned.dtypes.value_counts())

# %%
datetime_columns = df_cleaned.select_dtypes(include=["datetime64"]).columns

for col in datetime_columns:
    min_date = df_cleaned[col].min()
    df_cleaned[col] = (df_cleaned[col] - min_date).dt.days

print(df_cleaned.dtypes.value_counts())

# %%
df_cleaned.drop(columns=["Injury/Illness Category","Nature of Injury/Illness","Company Rating"],inplace=True)

# %%
df_cleaned["Weeks Open"].replace("(0)","0",inplace=True)
df_cleaned["Weeks Open"] = df_cleaned["Weeks Open"].astype(int)

df_cleaned["YOS"].replace("#NUM!",0,inplace=True)
df_cleaned["YOA"].replace("#NUM!",0,inplace=True)
df_cleaned["YOA"] = df_cleaned["YOA"].astype(int)
df_cleaned["YOS"] = df_cleaned["YOS"].astype(int)



# %%
import pprint
object_columns = df_cleaned.select_dtypes(include=["object"]).columns

encoding_strategies={}

for col in object_columns:
    if col == "Claim Decision":
        encoding_strategies[col] = "label_encoding"
    elif df_cleaned[col].nunique() <= 10:
        encoding_strategies[col] = "one_hot"
    elif df_cleaned[col].nunique() <= 50:
        encoding_strategies[col] = "label_encoding"
    else:
        encoding_strategies[col] = "binary"

pprint.pprint(encoding_strategies)



# %%




# %%
print(df_cleaned["Weeks Open"].dtype)

# %%
print(df_cleaned["Weeks Open"].value_counts())

# %%
print(df_cleaned["YOA"].dtype)

# %%
df_cleaned["Gender"].value_counts()

# %%
def transform_gender(gender):
    if gender in ["M","Male"]:
        return "Male"
    elif gender in ["F","Female"]:
        return "Female"
    else:
        return "Other"

df_cleaned["Gender"] = df_cleaned["Gender"].apply(transform_gender)
df_cleaned["Gender"].value_counts()


# %%
from sklearn.preprocessing import LabelEncoder
from category_encoders import BinaryEncoder

df_encoded = df_cleaned.copy()

for col,strategy in encoding_strategies.items():
    if strategy == "one_hot":
        dummies = pd.get_dummies(df_encoded[col],prefix=col)
        df_encoded = pd.concat([df_encoded,dummies],axis=1)
        df_encoded.drop(columns=[col],inplace=True)
    elif strategy == "label_encoding":
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    elif strategy == "binary":
        be = BinaryEncoder(cols = [col],drop_invariant=True)
        binary_encoded_cols = be.fit_transform(df_encoded[col])
        df_encoded.drop(columns=[col],inplace=True)
        df_encoded = pd.concat([df_encoded,binary_encoded_cols],axis=1)

print(df_encoded.info(verbose=True,show_counts=True))










# %%
bool_cols = df_encoded.select_dtypes(include=["bool"]).columns

for col in bool_cols:
    df_encoded[col] = df_encoded[col].astype(int)

print(df_encoded.info(verbose=True,show_counts=True))


# %%
df_encoded.to_csv("../data/data_xgboost_combined.csv",index=False)
df_encoded.head(10)


