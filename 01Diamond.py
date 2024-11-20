import pylab
import calendar
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import missingno as msno
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=DeprecationWarning)
df = pd.read_csv('./data/diamond.csv')
# df = df.fillna(df.mean(numeric_only=True))
# print(df.describe())

# msno.matrix(df, figsize=(12,9))
# plt.show()

x = df.loc[: , 'Carat Weight': 'Report']
# y = df['Price']
print(x.dtypes)
print(sorted(x['Cut'].unique()))
print(sorted(x['Color'].unique()))
print(sorted(x['Clarity'].unique()))
print(sorted(x['Polish'].unique()))
print(sorted(x['Symmetry'].unique()))
print(sorted(x['Report'].unique()))
# x['Cut'] = x['Cut'].str.strip().str.lower()  # Example for 'Cut' column
# x["Cut"] = x.Cut.map({1:'Fair', 2 :'Good', 3 :'Ideal', 4 :'Signature-Ideal', 5:'Very Good'})
# x['Cut'] = x['Cut'].astype(str)
# x["Color"] = x.Color.map({1: 'D', 2 : 'E', 3 : 'F', 4 : 'G', 5: 'H', 6: 'I'})
# x["Clarity"] = x.Clarity.map({1: 'FL', 2 : 'IF', 3 : 'VVS1', 4 : 'VVS2', 5: 'VS1', 6: 'VS2', 7: 'SI1'})
# x["Polish"] = x.Polish.map({1: 'Ex', 2 : 'G', 3 : 'ID', 4 : 'VG'})
# x["Symmetry"] = x.Symmetry.map({1: 'Ex', 2 : 'G', 3 : 'ID', 4 : 'VG'})
# x["Report"] = x.Report.map({1: 'AGSL', 2 : 'GIA'})
print(x.describe)
categoryVariableList= ['Carat Weight', 'Cut', 'Color', 'Clarity', 'Polish', 'Symmetry', 'Report']
# for var in categoryVariableList:
#     x[var] = x[var].astype("category")
X  =x.drop(["Carat Weight"],axis=1)

dataTypeDf = pd.DataFrame(X.dtypes.value_counts()).reset_index().rename(columns={"index":"variableType",0:"count"})

import itertools

categorical_features = ['Cut', 'Color', 'Clarity', 'Polish', 'Symmetry', 'Report']
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
colors = sns.color_palette("husl", len(categorical_features))

for i, feature in enumerate(categorical_features):
    ax = axes[i]
    sns.countplot(data=x, x=feature, ax=ax, color=colors[i])

    ax.set_title(f'Count of {feature}')
    ax.set_xlabel(feature)
    ax.set_ylabel('Count')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
plt.show()

'''
from sklearn.preprocessing import MinMaxScaler

# 수량 데이터를 포함하는 데이터 프레임 (예: count_df)
# MinMaxScaler 객체 생성
scaler = MinMaxScaler()


categorical_features = ['Cut', 'Color', 'Clarity', 'Polish', 'Symmetry', 'Report']

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

colors = sns.color_palette("husl", len(categorical_features))

for i, feature in enumerate(categorical_features):
    ax = axes[i]

    # 루프 내부에서 현재 특성에 대한 카운트를 포함하는 DataFrame 생성
    count_df = x.groupby(feature).size().reset_index(name='Count')

    # 데이터 변환
    count_df['Scaled_Count'] = scaler.fit_transform(count_df[['Count']])

    # 카운트 데이터를 사용하여 박스플롯 생성
    sns.boxplot(data=count_df, x=feature, y='Scaled_Count', ax=ax, color=colors[i])

    ax.set_title(f'Distribution of Counts by {feature} (Scaled)')
    ax.set_xlabel(feature)
    ax.set_ylabel('Scaled Count')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.show()

'''

# Step 1: Convert categorical features to numerical
categorical_mappings = {}
for feature in ['Cut', 'Color', 'Clarity', 'Polish', 'Symmetry', 'Report']:
    unique_values = sorted(df[feature].unique())  # Use original DataFrame 'df'
    categorical_mappings[feature] = {value: i + 1 for i, value in enumerate(unique_values)}
for feature, mapping in categorical_mappings.items():
    df[feature] = df[feature].map(mapping)  # Apply mapping to original DataFrame 'df'

# Step 2: Exclude 'Symmetry' and 'Report' from heatmap
heatmap_columns = ['Cut', 'Color', 'Clarity', 'Polish', 'Carat Weight', 'Price']
corr_matrix = df[heatmap_columns].corr()  # Use original DataFrame 'df'

# Step 3: Price as Key Factor (heatmap already uses it)
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix (Price as Key Factor)')
plt.show()