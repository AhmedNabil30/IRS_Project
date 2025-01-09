# Investment Options Recommendation Engine

## Project Overview
This project presents an **Investment Options Recommendation Engine** designed to offer personalized stock recommendations based on user preferences and company attributes. The system uses advanced machine learning techniques including collaborative filtering, matrix factorization, and content-based approaches.

## Features
- **User-Based Collaborative Filtering**
- **Item-Based Collaborative Filtering**
- **Matrix Factorization (PCA, SVD)**
- **Content-Based Recommender Systems**
- **Clustering-Based Methods**

## Dataset
The dataset contains 452 records with 476 features, including:
- **Financial Metrics**: Market capitalization, P/E ratio, revenue growth.
- **Trading Data**: Day high/low prices, volume, bid/ask data.
- **ESG Scores**: Environmental, social, and governance scores.
- **Analyst Ratings**: Buy, hold, or sell ratings from various financial analysts.

### Preprocessing Steps
1. **Handling Missing Values**: Dropped columns with more than 50% missing data, mean imputation for numeric columns, and 'unknown' for categorical missing values.
2. **Text Cleaning**: Removed special characters, applied proper casing, and stripped whitespace.
3. **Encoding**: Used label encoding and one-hot encoding for categorical variables.
4. **Normalization**: Applied Min-Max scaling to numeric features.
5. **Feature Engineering**: Used TF-IDF to convert text data into numerical features.
6. **Data Visualization**: Bar charts, scatter plots, heatmaps, histograms, box plots, KDE plots, and violin plots were used for analysis.

## Methodologies
### 1. Collaborative Filtering
#### User-Based Collaborative Filtering
- Computes similarity between users using cosine similarity, Pearson correlation, and adjusted cosine similarity.
- Predicts ratings based on similar users' preferences.
- Recommends top-N stocks with the highest predicted ratings.

#### Item-Based Collaborative Filtering
- Computes similarity between stocks based on features.
- Recommends stocks similar to those a user has interacted with.

### 2. Matrix Factorization
- **PCA with Mean-Filling**: Missing values are imputed with the mean, followed by PCA for dimensionality reduction.
- **PCA with MLE**: Iterative imputation followed by PCA.
- **SVD**: Singular Value Decomposition for latent factor extraction and rating prediction.

### 3. Content-Based Filtering
- Creates item profiles using features such as industry, market cap, dividend yield, ESG scores, and profit margins.
- Uses cosine similarity to recommend stocks based on feature similarity.

### 4. Clustering-Based Methods
- Uses K-Means clustering to group companies with similar profiles.
- Recommends stocks within the same cluster as a user-specified stock.

## Results
- **User-Based Collaborative Filtering**: Cosine similarity provided more optimistic recommendations compared to Pearson correlation and adjusted cosine methods.
- **Matrix Factorization**: PCA with MLE produced the highest predicted ratings.
- **Content-Based Filtering**: Recommended stocks with high similarity scores based on user preferences.

## Tools and Libraries
- **Programming Language**: Python
- **Libraries**: pandas, numpy, sklearn, scipy, matplotlib, seaborn

## Contributors
- **Marina Reda Abdullah Mekhael** (ID: 221101235)
- **Mohamed Mosad Fawzy** (ID: 221100611)

