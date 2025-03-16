# Machine Learning - Project

## Introductory Information

<img src="https://github.com/user-attachments/assets/9002855f-3f97-4b41-a180-85d1e24ad34a" alt="University Logo" width="150" align="right"/>

**University of Prishtina**  
**Faculty of Computer and Software Engineering**  
**Master’s Program**  
Course: **Machine Learning**  

## Course Professor

- **Prof. Lule Ahmedi**
- **Prof. Mergim Hoti**

## Project Team Members (Group 1)

- **Bleron Morina**
- **Endrit Gjoka**
- **Rukije Morina**

---
---
# Phase I

## Dataset Description

The chosen dataset is a merged version of two datasets:
[World Economic Indicators Dataset](https://www.kaggle.com/datasets/imtkaggleteam/mental-health/data?select=2-+burden-disease-from-each-mental-illness%281%29.csv) + [Mental Health Dataset](https://mavenanalytics.io/data-playground?order=date_added%2Cdesc&search=world%20economic%20indicator)

The merged dataset(Processed Dataset/FinalMerged.csv) contains 5510 rows and the 60 following columns:

| Index | Column                                                                                  | Dtype   |
|-------|-----------------------------------------------------------------------------------------|---------|
| 0     | Country Code                                                                            | string  |
| 1     | Contry Name                                                                             | string  |
| 2     | Region                                                                                  | string  |
| 3     | Year                                                                                    | int64   |
| 4     | abr                                                                                     | float64 |
| 5     | co2_prod                                                                                | float64 |
| 6     | coef_ineq                                                                               | float64 |
| 7     | diff_hdi_phdi                                                                           | float64 |
| 8     | eys                                                                                     | float64 |
| 9     | eys_f                                                                                   | float64 |
| 10    | eys_m                                                                                   | float64 |
| 11    | gdi                                                                                     | float64 |
| 12    | gii                                                                                     | float64 |
| 13    | gni_pc_f                                                                                | float64 |
| 14    | gni_pc_m                                                                                | float64 |
| 15    | gnipc                                                                                   | float64 |
| 16    | hdi                                                                                     | float64 |
| 17    | hdi_f                                                                                   | float64 |
| 18    | hdi_m                                                                                   | float64 |
| 19    | ihdi                                                                                    | float64 |
| 20    | ineq_edu                                                                                | float64 |
| 21    | ineq_inc                                                                                | float64 |
| 22    | ineq_le                                                                                 | float64 |
| 23    | le                                                                                      | float64 |
| 24    | le_f                                                                                    | float64 |
| 25    | le_m                                                                                    | float64 |
| 26    | lfpr_f                                                                                  | float64 |
| 27    | lfpr_m                                                                                  | float64 |
| 28    | loss                                                                                    | float64 |
| 29    | mf                                                                                      | float64 |
| 30    | mmr                                                                                     | int64   |
| 31    | mys                                                                                     | float64 |
| 32    | mys_f                                                                                   | float64 |
| 33    | mys_m                                                                                   | float64 |
| 34    | phdi                                                                                    | float64 |
| 35    | pr_f                                                                                    | float64 |
| 36    | pr_m                                                                                    | float64 |
| 37    | se_f                                                                                    | float64 |
| 38    | se_m                                                                                    | float64 |
| 39    | IncomeGroup                                                                             | object  |
| 40    | Birth rate, crude (per 1,000 people)                                                    | float64 |
| 41    | Death rate, crude (per 1,000 people)                                                    | float64 |
| 42    | Electric power consumption (kWh per capita)                                             | float64 |
| 43    | GDP (USD)                                                                               | float64 |
| 44    | GDP per capita (USD)                                                                    | float64 |
| 45    | Individuals using the Internet (% of population)                                        | float64 |
| 46    | Infant mortality rate (per 1,000 live births)                                           | float64 |
| 47    | Life expectancy at birth (years)                                                        | float64 |
| 48    | Population density (people per sq. km of land area)                                     | float64 |
| 49    | Unemployment (% of total labor force) (modeled ILO estimate)                            | float64 |
| 50    | DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Depressive disorders          | float64 |
| 51    | DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Schizophrenia                   | float64 |
| 52    | DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Bipolar disorder                | float64 |
| 53    | DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Eating disorders                | float64 |
| 54    | DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Anxiety disorders               | float64 |
| 55    | Schizophrenia disorders (share of population) - Sex: Both - Age: Age-standardized          | float64 |
| 56    | Depressive disorders (share of population) - Sex: Both - Age: Age-standardized           | float64 |
| 57    | Anxiety disorders (share of population) - Sex: Both - Age: Age-standardized              | float64 |
| 58    | Bipolar disorders (share of population) - Sex: Both - Age: Age-standardized              | float64 |
| 59    | Eating disorders (share of population) - Sex: Both - Age: Age-standardized               | float64 |


# Data Merging Process

## Step 1: Merging the Mental Health Dataset  
We first merged the two main data files from the **Mental Health Dataset**. This was done using an **INNER JOIN** based on the `Country Code` (or `Code` in some files) and `Year`. This ensured that only matching records from both files were included, maintaining data consistency.

## Step 2: Merging with the Socioeconomic Indicator Dataset  
After merging the Mental Health Dataset, we integrated it with the **World Socioeconomic Indicator Dataset**. This dataset contains various economic, social, and demographic indicators that provide deeper insights into mental health trends across different countries and years.

## Final Output  
The final dataset obtained from this process serves as the **complete output** from the Data Gathering and Preparation course. It is now ready for further analysis and insights extraction, ensuring that both **mental health statistics** and **socioeconomic indicators** are properly aligned and structured.

## SMOTE Balancing Script and Results Explanation

This script demonstrates the application of the SMOTE (Synthetic Minority Over-sampling Technique) algorithm to balance an imbalanced dataset. The process is part of a machine learning project where class imbalance can negatively impact model training and performance.

### What the Script Does

1. **Dataset Loading and Preparation:**  
   - The script loads the original dataset (5,510 rows) from a CSV file.
   - It converts categorical variables into dummy/indicator variables using `pd.get_dummies()`.
   - It automatically identifies the target column (if not explicitly named `"target"`, it uses the last column) for classification tasks.

2. **Train-Test Split:**  
   - The dataset is split into training and test sets. The training set is used for model training and balancing, while the test set remains untouched to ensure an unbiased evaluation.
   - For example, an 80/20 split would yield approximately 4,408 rows for training and 1,102 rows for testing.

3. **Visualizing Class Distribution Before SMOTE:**  
   - The script creates a plot to show the distribution of classes in the training set before applying SMOTE. This helps visualize the imbalance in the data.

4. **Applying SMOTE:**  
   - SMOTE is applied only on the training set. It generates synthetic samples for the minority class by interpolating between existing samples, thereby balancing the class distribution.
   - After applying SMOTE, the training set size increases (in our example, from around 4,408 rows to 6,404 rows), with each class now having an equal number of samples.

5. **Visualizing Class Distribution After SMOTE:**  
   - A second plot shows the balanced class distribution in the training set after SMOTE has been applied.

6. **Saving the Updated Dataset:**  
   - The balanced training set is saved as a new CSV file (`FinalMerged_balanced.csv`), so you can use it for further analysis or model training.

### Results and Interpretation

- **Before SMOTE:**  
  The training set has an imbalanced distribution, with one class having significantly fewer samples than the other. This imbalance can lead to biased model training.

- **After SMOTE:**  
  The SMOTE algorithm generates synthetic data points for the minority class, resulting in a balanced training set. In our example, both classes in the training set now have 3,202 samples, leading to a total of 6,404 rows in the balanced training set.

- **Important Note:**  
  The SMOTE process only affects the training data. The test set remains in its original form (approximately 1,102 rows), ensuring that model evaluation reflects the original data distribution.

Below is an example of the results visualization from the script:

<img width="1000" alt="image" src="https://github.com/user-attachments/assets/23809831-395d-4b3f-958e-9784f248983e" />

The image shows two side-by-side bar plots:
- The **left plot** displays the class distribution before applying SMOTE.
- The **right plot** shows the class distribution after SMOTE, where both classes are balanced.


