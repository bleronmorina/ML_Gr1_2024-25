import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Load the dataset (adjust the path if needed)
df = pd.read_csv('Processed Dataset/FinalMerged.csv')
print("Initial dataset shape:", df.shape)

# Convert categorical variables into dummy/indicator variables
df = pd.get_dummies(df, drop_first=True)

# Identify the target column; if 'target' is not present, use the last column
if 'target' not in df.columns:
    target_col = df.columns[-1]
    print(f"'target' column not found. Using '{target_col}' as the target variable.")
else:
    target_col = 'target'

# Split the dataset into features and target
X = df.drop(target_col, axis=1)
y = df[target_col]

# Split the data into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Visualize class distribution before applying SMOTE
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.countplot(x=y_train)
plt.title("Before SMOTE")
plt.xlabel("Class")
plt.ylabel("Count")

# Apply SMOTE to balance the training data
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print("Class distribution after SMOTE:")
print(pd.Series(y_train_sm).value_counts())

# Visualize class distribution after applying SMOTE
plt.subplot(1, 2, 2)
sns.countplot(x=y_train_sm)
plt.title("After SMOTE")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Save the balanced training dataset to a new CSV file (without scaling)
balanced_train_df = pd.DataFrame(X_train_sm, columns=X.columns)
balanced_train_df[target_col] = y_train_sm
balanced_train_df.to_csv('Processed Dataset/FinalMerged_balanced.csv', index=False)
print("Balanced training dataset saved to 'Processed Dataset/FinalMerged_balanced.csv'")
