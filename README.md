# Customer Segmentation Analysis

![Python](https://img.shields.io/badge/Python-3.9-blue)
![Pandas](https://img.shields.io/badge/Pandas-1.5.3-green)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.0-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 📊 Project Overview

A comprehensive customer segmentation analysis using RFM (Recency, Frequency, Monetary) analysis and K-Means clustering on online retail transaction data. This project identifies customer segments for targeted marketing strategies.

**Dataset**: Online Retail II (1+ million transactions, 2009-2011)
**Customers Analyzed**: 5,881
**Total Customer Value**: £13.4M

## 🎯 Business Problem

Businesses lose 15-30% of revenue from poor customer targeting. This project addresses:
- Identifying high-value customers for retention
- Detecting at-risk customers for win-back campaigns
- Segmenting customers for personalized marketing
- Optimizing marketing spend through data-driven insights

## 📁 Project Structure
customer-segmentation-analysis/
├── data/ # Data files (raw, cleaned, processed)
├── notebooks/ # Analysis scripts
│ ├── 01_exploration.py # Data exploration & understanding
│ ├── 02_cleaning.py # Data cleaning & preprocessing
│ ├── 03_rfm_analysis.py # RFM analysis & segmentation
│ ├── 04_clustering.py # K-Means clustering analysis
│ └── 05_visualization.py # Visualizations & business reporting
├── reports/ # Analysis reports
│ ├── images/ # Visualizations & charts
│ ├── final/ # Final business reports
│ └── *.txt # Detailed analysis reports
├── requirements.txt # Python dependencies
└── README.md

## 🔍 Methodology

### 1. **Data Exploration**
- Loaded 1,067,371 transactions from Excel sheets
- Identified data quality issues: 22.8% missing Customer IDs
- Analyzed transaction patterns and business metrics

### 2. **Data Cleaning**
- Removed transactions without Customer IDs (critical for segmentation)
- Handled returns (negative quantities) separately
- Capped outliers in Price and Quantity using IQR method
- Result: Cleaned dataset of 805,620 transactions

### 3. **RFM Analysis**
- **Recency**: Days since last purchase
- **Frequency**: Number of unique purchases
- **Monetary**: Total customer spend
- Created 7 customer segments using business rules

### 4. **Machine Learning Clustering**
- Applied K-Means clustering on RFM features
- Standardized features and used log transformation for Monetary
- Identified 4 natural customer clusters
- Compared with RFM segments for validation

### 5. **Business Insights & Visualization**
- Created professional visualizations for stakeholders
- Developed actionable recommendations per segment
- Generated executive dashboard and business report

## 📈 Key Findings

### Customer Segments Identified:

| Segment | Customers | Revenue Contribution | Action Required |
|---------|-----------|---------------------|-----------------|
| **Champions** | 1,826 (31.0%) | £10.1M (75.4%) | Premium loyalty programs |
| **At Risk** | 1,692 (28.8%) | £2.2M (16.2%) | Win-back campaigns |
| **Hibernating** | 1,235 (21.0%) | £0.5M (3.5%) | Reactivation efforts |
| **New Customers** | 337 (5.7%) | £0.1M (0.9%) | Onboarding series |

### High-Value Insights:
- **Top 21 customers** (0.4% of base) generate **£1.6M** in revenue
- **Champions segment** drives **75.4%** of total revenue
- **At-risk customers** represent **£2.2M** in potential churn

## 📊 Visualizations

![RFM Analysis](reports/images/06_rfm_analysis.png)
*RFM Segment Analysis*

![Business Dashboard](reports/images/09_business_dashboard.png)
*Executive Business Dashboard*

![Cluster Analysis](reports/images/07_cluster_analysis.png)
*Machine Learning Clusters*

## 🛠️ Technical Implementation

### Dependencies
```bash
pandas==1.5.3
numpy==1.24.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.17.0