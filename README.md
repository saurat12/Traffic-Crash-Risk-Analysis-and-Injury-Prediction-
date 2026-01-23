# Traffic Crash Risk Analysis and Injury Prediction

A data science project analyzing traffic crash patterns and predicting injury severity using machine learning techniques on Cincinnati Police Department crash data.

## Project Overview

This project analyzes traffic crash data to identify risk patterns and build predictive models for crash severity. Using real-world crash report data from Cincinnati (through April 2025), the analysis combines exploratory data analysis, statistical methods, clustering techniques, and logistic regression to understand crash dynamics and predict injury outcomes.

## Dataset

- **Source**: Cincinnati Police Department Crash Reports (CPD_Crash_Reports_CPD_20250420.csv)
- **Records**: 392,949 observations after data cleaning
- **Features**: 28 variables including:
  - Location data (latitude, longitude, neighborhood, ZIP code)
  - Temporal data (crash date, day of week)
  - Environmental conditions (weather, road conditions, road surface, road contour)
  - Crash characteristics (manner of crash, crash severity, unit type)
  - Person characteristics (age, gender, type of person, injuries)

## Key Features

### 1. Data Preprocessing
- Handling missing values and data type conversions
- Outlier detection and removal using Tukey's fence method (IQR rule)
- Feature engineering for analysis
- Final cleaned dataset: 342,566 records

### 2. Exploratory Data Analysis
- Age distribution analysis (853 outliers identified)
- Crash severity distribution
- Temporal patterns (day of week analysis)
- Geographic distribution across Cincinnati neighborhoods

### 3. Machine Learning Models

#### Logistic Regression for Injury Prediction
- **Target variable**: Binary injury indicator (injured vs. not injured)
- **Features**: Age, day of week
- **Performance**:
  - Test Set Accuracy: 71.6%
  - ROC AUC Score: 0.542
  - Model equation: `logit(P(Injury)) = -0.5717 - 0.0091×AGE - 0.0055×DAY_OF_WEEK`

#### K-Means Clustering
- **Purpose**: Identify crash pattern groups
- **Optimal k**: 2 clusters (Silhouette score: 0.650)
- **Features**: Age, day of week
- **Results**:
  - Cluster 0: 1,405 members
  - Cluster 1: 595 members

## Technologies Used

- **Python 3.x**
- **Libraries**:
  - `pandas`: Data manipulation and analysis
  - `numpy`: Numerical computing
  - `matplotlib` & `seaborn`: Data visualization
  - `scikit-learn`: Machine learning algorithms
    - LogisticRegression
    - KMeans
    - StandardScaler
    - train_test_split
  - `scipy`: Hierarchical clustering

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/traffic-crash-analysis.git
cd traffic-crash-analysis

# Install required packages
pip install pandas numpy matplotlib seaborn scikit-learn scipy
