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

| Index | Column                                                                            | Dtype   |
| ----- | --------------------------------------------------------------------------------- | ------- |
| 0     | Country Code                                                                      | string  |
| 1     | Contry Name                                                                       | string  |
| 2     | Region                                                                            | string  |
| 3     | Year                                                                              | int64   |
| 4     | abr                                                                               | float64 |
| 5     | co2_prod                                                                          | float64 |
| 6     | coef_ineq                                                                         | float64 |
| 7     | diff_hdi_phdi                                                                     | float64 |
| 8     | eys                                                                               | float64 |
| 9     | eys_f                                                                             | float64 |
| 10    | eys_m                                                                             | float64 |
| 11    | gdi                                                                               | float64 |
| 12    | gii                                                                               | float64 |
| 13    | gni_pc_f                                                                          | float64 |
| 14    | gni_pc_m                                                                          | float64 |
| 15    | gnipc                                                                             | float64 |
| 16    | hdi                                                                               | float64 |
| 17    | hdi_f                                                                             | float64 |
| 18    | hdi_m                                                                             | float64 |
| 19    | ihdi                                                                              | float64 |
| 20    | ineq_edu                                                                          | float64 |
| 21    | ineq_inc                                                                          | float64 |
| 22    | ineq_le                                                                           | float64 |
| 23    | le                                                                                | float64 |
| 24    | le_f                                                                              | float64 |
| 25    | le_m                                                                              | float64 |
| 26    | lfpr_f                                                                            | float64 |
| 27    | lfpr_m                                                                            | float64 |
| 28    | loss                                                                              | float64 |
| 29    | mf                                                                                | float64 |
| 30    | mmr                                                                               | int64   |
| 31    | mys                                                                               | float64 |
| 32    | mys_f                                                                             | float64 |
| 33    | mys_m                                                                             | float64 |
| 34    | phdi                                                                              | float64 |
| 35    | pr_f                                                                              | float64 |
| 36    | pr_m                                                                              | float64 |
| 37    | se_f                                                                              | float64 |
| 38    | se_m                                                                              | float64 |
| 39    | IncomeGroup                                                                       | object  |
| 40    | Birth rate, crude (per 1,000 people)                                              | float64 |
| 41    | Death rate, crude (per 1,000 people)                                              | float64 |
| 42    | Electric power consumption (kWh per capita)                                       | float64 |
| 43    | GDP (USD)                                                                         | float64 |
| 44    | GDP per capita (USD)                                                              | float64 |
| 45    | Individuals using the Internet (% of population)                                  | float64 |
| 46    | Infant mortality rate (per 1,000 live births)                                     | float64 |
| 47    | Life expectancy at birth (years)                                                  | float64 |
| 48    | Population density (people per sq. km of land area)                               | float64 |
| 49    | Unemployment (% of total labor force) (modeled ILO estimate)                      | float64 |
| 50    | DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Depressive disorders    | float64 |
| 51    | DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Schizophrenia           | float64 |
| 52    | DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Bipolar disorder        | float64 |
| 53    | DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Eating disorders        | float64 |
| 54    | DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Anxiety disorders       | float64 |
| 55    | Schizophrenia disorders (share of population) - Sex: Both - Age: Age-standardized | float64 |
| 56    | Depressive disorders (share of population) - Sex: Both - Age: Age-standardized    | float64 |
| 57    | Anxiety disorders (share of population) - Sex: Both - Age: Age-standardized       | float64 |
| 58    | Bipolar disorders (share of population) - Sex: Both - Age: Age-standardized       | float64 |
| 59    | Eating disorders (share of population) - Sex: Both - Age: Age-standardized        | float64 |

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
In Phase II, we implemented and evaluated four supervised machine learning algorithms—Logistic Regression, Random Forest, Gradient Boosting, and SVM with RBF kernel—to classify countries into five levels based on the share of schizophrenia disorders. The continuous target variable was discretized into quintiles to represent severity levels. After preprocessing and scaling the data, models were trained and compared using accuracy, precision, recall, F1-score, and confusion matrices. Tree-based models (Random Forest and Gradient Boosting) outperformed the others, indicating complex non-linear interactions among predictors.

---

# Supervised Algorithms for Multi-Level Mental Health Indicator Classification

**Overview**
In this phase, we train and evaluate four supervised classifiers on our **FinalMerged.csv** dataset. The goal is to classify countries/years into multiple levels (typically 5, from very low to very high) based on a user-selected mental health indicator.

The dataset contains:

*   **Demographic/economic indicators** (e.g. GDP per capita, life expectancy, HDI, electricity consumption)
*   **Mental-health metrics** (e.g. age-standardized share of various disorders, DALYs rates for various causes)
*   **Temporal and categorical fields** (Country, Region, Year)

**Target Variable Selection and Preparation:**
For this run, the selected target variable was:

*   `Schizophrenia disorders (share of population) - Sex: Both - Age: Age-standardized`

This continuous target variable was **divided into 5 levels (quintiles)** using `pandas.qcut`. This created a multi-class target variable labeled `0` (lowest 20%) to `4` (highest 20%), representing categories from 'Very Low' to 'Very High' share for Schizophrenia disorders. The class distribution was balanced (20% in each class). All predictor features used are numeric and undergo median imputation for missing values and standard scaling.

---

## Why These Algorithms?

These algorithms are chosen for their different strengths in handling classification tasks, including multi-class problems:

### 1. Logistic Regression

*   **What it does:** Models the probability of each class using a logistic function over a linear combination of features. For multi-class, it typically uses a One-vs-Rest (OvR) or Multinomial approach.
*   **Why it fits:**
    *   Provides a strong, interpretable baseline. Coefficients can indicate how features relate to the log-odds of belonging to different classes (relative to others).
    *   Efficient and works well when relationships are approximately linear.

### 2. Random Forest

*   **What it does:** An ensemble of decision trees. Each tree votes, and the majority class wins. Handles multi-class natively.
*   **Why it fits:**
    *   Excellent at capturing **non-linear interactions** between features (e.g., how high unemployment *and* low HDI together might predict a specific level of the mental health indicator).
    *   Robust to feature scaling.
    *   Provides **feature importance** scores, indicating which socio-economic factors are most influential in distinguishing between the different levels of the target indicator.

### 3. Gradient Boosting

*   **What it does:** Builds trees sequentially, with each new tree correcting errors made by the previous ones. Handles multi-class effectively.
*   **Why it fits:**
    *   Often achieves high predictive accuracy on structured data.
    *   Includes regularization techniques to prevent overfitting.
    *   Can implicitly handle complex feature interactions.

### 4. Support Vector Machine (RBF Kernel)

*   **What it does:** Finds optimal separating hyperplanes between classes in a high-dimensional space created by the kernel function. Handles multi-class typically via One-vs-One or One-vs-Rest strategies.
*   **Why it fits:**
    *   Effective when there are clear (though potentially non-linear) margins of separation between classes, especially after data standardization.
    *   The RBF kernel allows for flexible, complex decision boundaries.

---

## Model Performance

*(Note: Precision, Recall, and F1 Score are calculated using the 'weighted' average to account for class imbalance across the 5 levels)*

| Algorithm              | Accuracy | Precision (Weighted) | Recall (Weighted) | F1 Score (Weighted) |
| :--------------------- | -------: | -------------------: | ----------------: | ------------------: |
| Logistic Regression    |   0.9156 |               0.9172 |            0.9156 |              0.9159 |
| Random Forest          |   0.9782 |               0.9785 |            0.9782 |              0.9782 |
| Gradient Boosting      |   0.9746 |               0.9750 |            0.9746 |              0.9746 |
| Support Vector Machine |   0.9220 |               0.9240 |            0.9220 |              0.9221 |

---

## Confusion Matrices

*(Matrices show performance on the test set for classifying Schizophrenia disorder share into 5 quintiles (0=Lowest, 4=Highest))*

### Logistic Regression

|           | Pred 0 | Pred 1 | Pred 2 | Pred 3 | Pred 4 |
| :-------- | -----: | -----: | -----: | -----: | -----: |
| **Actual 0** |    215 |      6 |      0 |      0 |      0 |
| **Actual 1** |     11 |    199 |     10 |      0 |      0 |
| **Actual 2** |      0 |      5 |    190 |     25 |      0 |
| **Actual 3** |      0 |      0 |     10 |    200 |     11 |
| **Actual 4** |      0 |      0 |      0 |     15 |    205 |

---

### Random Forest

|           | Pred 0 | Pred 1 | Pred 2 | Pred 3 | Pred 4 |
| :-------- | -----: | -----: | -----: | -----: | -----: |
| **Actual 0** |    218 |      3 |      0 |      0 |      0 |
| **Actual 1** |      3 |    217 |      0 |      0 |      0 |
| **Actual 2** |      0 |      4 |    211 |      5 |      0 |
| **Actual 3** |      0 |      0 |      2 |    217 |      2 |
| **Actual 4** |      0 |      0 |      0 |      5 |    215 |

---

### Gradient Boosting

|           | Pred 0 | Pred 1 | Pred 2 | Pred 3 | Pred 4 |
| :-------- | -----: | -----: | -----: | -----: | -----: |
| **Actual 0** |    218 |      3 |      0 |      0 |      0 |
| **Actual 1** |      3 |    217 |      0 |      0 |      0 |
| **Actual 2** |      0 |      5 |    208 |      7 |      0 |
| **Actual 3** |      0 |      0 |      2 |    216 |      3 |
| **Actual 4** |      0 |      0 |      0 |      5 |    215 |

---

### Support Vector Classifier (SVC)

|           | Pred 0 | Pred 1 | Pred 2 | Pred 3 | Pred 4 |
| :-------- | -----: | -----: | -----: | -----: | -----: |
| **Actual 0** |    219 |      2 |      0 |      0 |      0 |
| **Actual 1** |     11 |    189 |     20 |      0 |      0 |
| **Actual 2** |      0 |      6 |    200 |     14 |      0 |
| **Actual 3** |      0 |      0 |     13 |    202 |      6 |
| **Actual 4** |      0 |      0 |      1 |     13 |    206 |

---

## Discussion

*   **Model Comparison:** For classifying the share of Schizophrenia disorders into quintiles, Random Forest and Gradient Boosting significantly outperformed Logistic Regression and SVC, achieving accuracies and weighted F1-scores around 97-98%. Logistic Regression and SVC performed reasonably well (around 91-92%) but showed more confusion between adjacent classes, particularly visible in the off-diagonal elements of their confusion matrices (e.g., misclassifying Actual 1 as Pred 2, or Actual 3 as Pred 2).
*   **Interpretability:**
    *   The tree-based models (RF, GB) likely captured complex, non-linear relationships between the socio-economic predictors and the different levels of Schizophrenia burden more effectively than the linear model (LR) or the kernel-based SVM. Feature importances from RF/GB would be valuable to identify the key drivers distinguishing these levels.
*   **Algorithm Suitability:** The superior performance of the ensemble tree methods suggests that non-linearities and feature interactions are important for accurately categorizing Schizophrenia disorder shares based on the available predictors.

## Next Steps

1.  **Hyperparameter Tuning:**
    *   Use `GridSearchCV` or `RandomizedSearchCV` to optimize parameters for each model (e.g., `C` for Logistic Regression and SVC, `n_estimators` and `max_depth` for RF/GB, `gamma` for SVC) to potentially improve performance on the multi-class task, especially for LR and SVC.
2.  **Advanced Interpretability:**
    *   For tree models, delve deeper into feature importances.
    *   Consider using techniques like SHAP (SHapley Additive exPlanations) to understand feature contributions for individual predictions across the different classes.
3.  **Regression Experimentation:**
    *   Revert to using the original, continuous target column (`Schizophrenia disorders (share of population)...`) and train regression models (e.g., Linear Regression, RandomForestRegressor, GradientBoostingRegressor) to predict the actual share value instead of classifying into levels. This provides more granular predictions.
4.  **Target Variable Analysis:**
    *   Experiment with different numbers of bins (e.g., 3 levels, 10 levels) or different binning strategies (e.g., equal-width bins instead of quantiles) for the Schizophrenia share to see how it affects model performance and interpretation.
# Unsupervised Algorithms

**Overview**  
In this phase, we explore unsupervised learning techniques to uncover hidden patterns in the data without relying on labeled outcomes. Our aim is to group countries or observations with similar socioeconomic and mental-health profiles using **clustering algorithms**, and to reduce data dimensionality for effective **visualization** using **PCA**, **t-SNE**, and **UMAP**.

We work with the same preprocessed dataset used in the supervised phase, with all features scaled and encoded, and without any missing values.

---

## Why Use Unsupervised Learning?

- **Clustering** helps identify natural groupings among countries, such as those with similar economic conditions and mental-health burdens.
- **Dimensionality reduction** aids in visualizing high-dimensional relationships in 2D or 3D plots.
- It complements supervised learning by revealing structure in the data that may guide feature engineering or hypothesis generation.

---

## Algorithms Used

### 1. K-Means Clustering

- **What it does:**  
  Partitions the data into `k` clusters by minimizing intra-cluster variance.
- **Why we used it:**
  - Simple and fast.
  - Works well when clusters are roughly spherical and similarly sized.
  - Provides clear, hard cluster assignments.

### 2. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

- **What it does:**  
  Groups data points based on density. Points in low-density regions are labeled as noise.
- **Why we used it:**
  - Can find arbitrarily shaped clusters.
  - Robust to outliers.
  - Does not require specifying the number of clusters.

### 3. Agglomerative Hierarchical Clustering

- **What it does:**  
  Builds a hierarchy of clusters using a bottom-up approach by repeatedly merging the two closest clusters.
- **Why we used it:**
  - Captures nested structure in the data.
  - Flexible linkage criteria (e.g. average, complete, single).
  - Visualizable via dendrograms.

---

## Dimensionality Reduction

### 1. PCA (Principal Component Analysis)

- **What it does:**  
  Projects data into orthogonal components that explain maximum variance.
- **Why we used it:**
  - Useful for initial exploratory analysis.
  - Fast and interpretable.
  - Can be combined with clustering to visualize groupings.

### 2. t-SNE (t-distributed Stochastic Neighbor Embedding)

- **What it does:**  
  Embeds high-dimensional data in lower dimensions by preserving local structure.
- **Why we used it:**
  - Excellent for visualizing complex data.
  - Reveals subtle structure and clusters not visible with PCA.

### 3. UMAP (Uniform Manifold Approximation and Projection)

- **What it does:**  
  Constructs a high-dimensional graph and optimizes its low-dimensional layout.
- **Why we used it:**
  - Balances local and global structure preservation.
  - Generally faster and more scalable than t-SNE.
  - Produces more stable and meaningful embeddings.

---

## Results and Insights

### K-Means Clustering Results

The K-Means algorithm was applied with different numbers of clusters ranging from 3 to 6. The best silhouette score was achieved with 4 clusters. The silhouette score helps assess the quality of the clusters, with higher values indicating better-defined and more separated clusters.

- **Silhouette Scores for Different Cluster Numbers**
  -3 clusters: Silhouette Score = 0.1773
  -4 clusters: Silhouette Score = 0.1826 (Best score)
  -5 clusters: Silhouette Score = 0.1467
  -6 clusters: Silhouette Score = 0.1657

The best silhouette score was achieved with 4 clusters, indicating that this number best captures the structure in the data. Higher values of clusters resulted in a decrease in the silhouette score, suggesting overfitting or misinterpretation of the data.

### DBSCAN and Agglomerative Clustering Results

-DBSCAN: The Silhouette score was calculated only for valid clusters, and the Davies-Bouldin score was also computed. However, DBSCAN is better suited for finding arbitrarily shaped clusters and identifying noise points, which is particularly useful in the case of outliers.

-Agglomerative Clustering: This method produced hierarchical clusters, with the silhouette score indicating reasonable clustering quality. The Davies-Bouldin score was also computed for this method.

---

## Visualization

- Clusters discovered by K-Means and DBSCAN were visualized using PCA, t-SNE, and UMAP.
- In PCA space, countries with high HDI and low depression burden tended to group together.
- t-SNE and UMAP uncovered more fine-grained clusters, some of which aligned with geographic regions or income levels.
- DBSCAN was effective in identifying small, high-density subgroups and labeling noise points.

The following visualizations help us interpret the results of dimensionality reduction and clustering:

### PCA Visualization
The PCA plot below shows how the data is distributed across the first two principal components after K-Means clustering with 4 clusters. Each color represents a different cluster, showing clear separation between the clusters.

![PCA Clusters Plot](Processing%20Scripts/data_analysis/pca_clusters_plot_pca3.png)

### t-SNE Visualization

The t-SNE plot reveals how data points group together in lower dimensions, making the clusters more distinct. The t-SNE technique is particularly useful for complex datasets that PCA cannot visualize effectively.

![t-SNE Clusters Plot](Processing%20Scripts/data_analysis/pca_clusters_plot_tsne3.png)

### UMAP Visualization

UMAP provides a clear view of the clusters with respect to both local and global structures. It is faster and more scalable than t-SNE and offers an effective visualization for large datasets.

![t-SNE Clusters Plot](Processing%20Scripts/data_analysis/pca_clusters_plot_umap3.png)

### Silhouette Score vs. Number of Clusters

The plot below shows the Silhouette Score as a function of the number of clusters. The highest score is achieved with 4 clusters, confirming that this is the optimal number for the dataset.

![t-SNE Clusters Plot](Processing%20Scripts/data_analysis/silhouetteScore_NumberClusters.png)


---

## Next Steps

1. **Cluster Profiling**  
   – Summarize the characteristics of each cluster by examining median values of socioeconomic and health features.

2. **Temporal Analysis**  
   – Apply clustering separately by year to track how country groupings change over time.

3. **Use Clusters as Features**  
   – Feed cluster labels into supervised models to assess if cluster membership improves predictive performance.
