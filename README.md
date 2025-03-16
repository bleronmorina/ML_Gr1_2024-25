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
