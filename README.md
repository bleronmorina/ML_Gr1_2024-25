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

## Chosen Datasets  

For this machine learning course project, we selected two key datasets that provide valuable insights into mental health and socioeconomic conditions across different countries and years. These datasets are:  

### 1. **Mental Health Dataset**  
- This dataset, compiled by Saloni Dattani and updated by Mohamadreza Momeni, focuses on **mental health statistics**.  
- It includes data on **mental illness prevalence, survey-based mental health assessments, and diagnostic trends**.  
- The dataset is valuable because mental health significantly impacts quality of life, productivity, and social well-being.  
- However, it also has limitations, such as self-reporting biases and differences in mental health awareness across countries.  

### 2. **World Economic Indicators Dataset**  
- This dataset, sourced from the **World Bank and the United Nations**, contains economic and developmental indicators from **1960 to 2021**.  
- It includes key metrics such as **GDP per capita, life expectancy, electricity consumption, and the Human Development Index (HDI)**.  
- These indicators provide a comprehensive view of a country's **economic and social well-being**, making them essential for understanding the external factors affecting mental health.  

## Why These Datasets?  

We chose these datasets because they complement each other, allowing us to analyze the **relationship between economic conditions and mental health trends**. By merging them, we can:  

1. **Examine the Impact of Economic Factors on Mental Health:**  
   - Analyze how GDP per capita, employment rates, or education levels correlate with mental health conditions.  

2. **Identify Trends Across Countries and Years:**  
   - Compare mental health statistics in high-income vs. low-income nations.  
   - Investigate how economic development influences mental health over time.  

3. **Enhance the Predictive Power for Machine Learning Models:**  
   - By combining both datasets, we create a **more comprehensive feature set** for predictive modeling.  
   - This allows us to explore potential machine learning models that predict mental health trends based on socioeconomic indicators.  

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

## Missing Values Analysis
A comprehensive analysis of missing values was performed to ensure data completeness and integrity. Utilizing Python’s pandas library, all columns were systematically examined for null values.

### Results:
 - No missing values detected across the dataset.
The dataset is fully complete, eliminating the need for imputation or additional preprocessing related to missing data.
This guarantees that all features are available for further analysis, visualization, and model training without concerns regarding data gaps.


## Outlier Detection Analysis
A comprehensive analysis of outliers was performed to enhance data quality and ensure reliable results. Utilizing **Z-Score** and **DBSCAN clustering**, outliers were systematically identified and removed.

### Methodology:
 - **Z-Score Method:** Identified outliers with absolute Z-scores greater than `3`, indicating extreme deviations from the mean.
 - **DBSCAN Clustering:** Detected outliers using a density-based approach (`eps=0.7`, `min_samples=3`).

### Results:
 - **Total rows removed:** 1,818  
 - **Total rows flagged as outliers:** 1,818  
 - **Rows remaining after outlier removal:** 3,692  
 - **Cleaned dataset saved to:** `../../Processed Dataset/dataset_cleaned_03.csv`

The cleaned dataset contains only relevant data points, improving overall consistency for further analysis, visualization, and modeling.

### Distribution Comparisons:
To evaluate the impact of outlier removal, Kernel Density Estimation (KDE) plots were generated for selected features.
Below are two examples of the results visualization from the script:

![CO₂ Production Distribution](Processing%20Scripts/outliers/results/c02_prod_distribution.png)
![Life Expectancy Distribution](Processing%20Scripts/outliers/results/lifeexpectancy.png)


These visualizations illustrate the changes in distributions before and after outlier removal.

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


---
---
# Phase II

