# Productivity- and Season-based Agricultural Crop Recommendation Engine

## Project Overview

This project is an agricultural crop recommendation system that combines **content-based filtering** and **classification techniques** to suggest the most suitable crops. By analyzing soil nutrients, weather conditions, and seasonal data, the system assists farmers in maximizing productivity and optimizing decision-making.

---

## Features

### Intelligent Crop Recommendations
- **Technique Used**: Content-Based Filtering
  - Recommends crops by analyzing soil nutrients (Nitrogen, Phosphorus, Potassium), pH, and weather conditions (temperature, humidity, rainfall).
  - Utilizes environmental and seasonal data to ensure accurate recommendations.

### Productivity Enhancement
- Suggests strategies to maximize yield based on historical crop production data.

### Evaluation and Insights
- Evaluates the recommendation system's performance using precision and recall.
- Visualizes recommendations and predictions for user-friendly insights.

---

## Datasets

The following datasets were used in this project:

### Crop Recommendation Dataset
- Contains data on soil nutrients (Nitrogen, Phosphorus, Potassium), temperature, rainfall, pH, and humidity.
- **Download Link**: [Crop Recommendation Dataset](https://www.kaggle.com/code/niteshhalai/crop-recommendation-dataset)

### Crop Production Dataset
- Includes historical crop production data by state, district, season, and crop type.
- **Download Link**: [Crop Production Dataset](https://www.kaggle.com/datasets/abhinand05/crop-production-in-india)

---

## Model Details

### Crop Recommendation Model
- **Technique Used**: Content-Based Filtering
- **Algorithm**: Random Forest Classifier
- **Features**:
  - Soil nutrients: N, P, K
  - Weather conditions: Temperature, Humidity, pH, Rainfall
  - Geographical and seasonal data
- **Outputs**: The best crop recommendation for the given inputs.

### Yield Prediction Model
- **Technique Used**: Regression Analysis
- **Algorithm**: Linear Regression
- **Features**:
  - Land area, season, and historical production data
- **Outputs**: Predicted yield in kilograms per hectare.

---

## Outputs

### Crop Recommendations
- Example output: ["Rice", "Wheat", "Maize", "Barley"]
- These crops are recommended based on the input conditions provided.

### Model Evaluation
- **Crop Recommendation Model Accuracy**: 95.6%
- **Yield Prediction Model R² Score**: 89.4%

---

## Results

The system successfully:
- Recommends suitable crops based on soil and environmental parameters using content-based filtering.
- Predicts crop yield accurately for better planning using regression techniques.
- Visualizes data and insights for better decision-making.

---

## Team Members

- **Osama Gamal Hamed Ebraheem** - B20000007  
- **Sohila Mohamed Ali** - A20001134 
