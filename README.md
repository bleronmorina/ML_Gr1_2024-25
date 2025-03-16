# Machine Learning - Project

The chosen dataset is a merged version of two datasets:
[World Economic Indicators Dataset](https://www.kaggle.com/datasets/imtkaggleteam/mental-health/data?select=2-+burden-disease-from-each-mental-illness%281%29.csv) + [Mental Health Dataset](https://mavenanalytics.io/data-playground?order=date_added%2Cdesc&search=world%20economic%20indicator)
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

# Data Merging Process

## Step 1: Merging the Mental Health Dataset  
We first merged the two main data files from the **Mental Health Dataset**. This was done using an **INNER JOIN** based on the `Country Code` (or `Code` in some files) and `Year`. This ensured that only matching records from both files were included, maintaining data consistency.

## Step 2: Merging with the Socioeconomic Indicator Dataset  
After merging the Mental Health Dataset, we integrated it with the **World Socioeconomic Indicator Dataset**. This dataset contains various economic, social, and demographic indicators that provide deeper insights into mental health trends across different countries and years.

## Final Output  
The final dataset obtained from this process serves as the **complete output** from the Data Gathering and Preparation course. It is now ready for further analysis and insights extraction, ensuring that both **mental health statistics** and **socioeconomic indicators** are properly aligned and structured.
