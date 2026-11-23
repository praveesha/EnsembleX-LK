# EnsembleX LK
### An Explainable Ensemble Learning Framework for Sri Lankan Used Vehicle Price Prediction

<p align="center">
  <strong>Machine Learning · Ensemble Methods · Explainable AI · Streamlit</strong>
</p>

<p align="center">
  <a href="https://ensemblex-lk.streamlit.app/">Live Dashboard</a> ·
  <a href="https://github.com/praveesha/EnsembleX-LK">GitHub Repository</a> ·
  <a href="https://www.kaggle.com/datasets/prasadnirmal/srilankan-second-vehiclecar-price-dataset">Dataset</a>
</p>

---

## 1. Project Overview

**EnsembleX LK** is an explainable machine learning project that investigates the application of ensemble learning techniques to used vehicle price prediction in the Sri Lankan automotive market.

The project develops and evaluates multiple machine learning models to estimate vehicle asking prices based on attributes such as brand, model, year of manufacture, engine capacity, mileage, transmission type, fuel type, and other vehicle characteristics.

Rather than relying on a single predictive algorithm, the framework explores different ensemble strategies, including **Bagging, Boosting, Voting, and Stacking**, to examine their predictive performance and generalisation capabilities.

The project also incorporates model explainability through permutation feature importance, providing insights into the vehicle characteristics that contribute most to predictive performance.

The resulting model is integrated into an interactive Streamlit dashboard, allowing users to explore model comparisons, examine feature importance, and generate vehicle price estimates.

## 2. Project Objectives

The primary objectives are to:

- Develop a machine learning framework for predicting used vehicle asking prices in Sri Lanka.
- Investigate and compare multiple ensemble learning strategies.
- Evaluate predictive performance using established regression metrics.
- Identify the most effective model configurations for the selected dataset.
- Interpret model behaviour using explainability techniques.
- Develop an interactive dashboard to demonstrate the practical application of the trained model.

## 3. Dataset

The project uses the **Sri Lankan Second Vehicle/Car Price Dataset**, obtained from Kaggle.

**Dataset source:**  
https://www.kaggle.com/datasets/prasadnirmal/srilankan-second-vehiclecar-price-dataset

| Dataset characteristic | Description |
|---|---|
| Records | 9,788 |
| Original columns | 17 |
| Predictive target | `Price` |
| Target type | Continuous numerical variable |
| Market | Sri Lanka |
| Data characteristics | Numerical and categorical |

The dataset contains vehicle-related attributes, including:

- **Vehicle identification:** Brand, Model
- **Technical specifications:** Engine capacity, Gear, Fuel Type
- **Vehicle history:** Year of Manufacture (YOM), Mileage
- **Market information:** Town, Listing Date, Leasing
- **Vehicle features:** Condition, Air Conditioning, Power Steering, Power Mirror, Power Window
- **Target variable:** Price

The original `Unnamed: 0` index field is excluded from predictive modelling.

### Data validation

The dataset was subjected to initial quality checks:

- No missing values were identified.
- No duplicate records were identified.
- The Kaggle dataset was verified against the previously used CSV, confirming identical dimensions, column ordering, and values.

**Target variable:** The `Price` field represents vehicle asking prices in the dataset's original units, expressed in LKR lakhs. These values should not be interpreted as verified transaction prices.

## 4. Methodology

The project follows a structured machine learning workflow.

### 4.1 Exploratory Data Analysis

Exploratory analysis is conducted to understand:

- Vehicle price distributions and variability.
- Brand and model representation.
- Numerical feature distributions.
- Categorical feature frequencies.
- Relationships between vehicle characteristics and asking prices.

### 4.2 Data Preprocessing and Feature Engineering

The preprocessing workflow includes:

- Removing non-predictive index information.
- Converting listing dates into appropriate datetime representations.
- Handling numerical and categorical variables through preprocessing pipelines.
- Encoding categorical features.
- Deriving additional temporal and vehicle-related features, including:
  - Vehicle Age
  - Listing Year
  - Listing Month

The preprocessing pipeline is integrated with the predictive models to maintain consistency between training and inference.

### 4.3 Ensemble Learning

The following approaches are implemented and evaluated:

| Model | Learning approach |
|---|---|
| Decision Tree | Individual tree-based baseline |
| Random Forest | Bagging |
| Extra Trees | Randomised tree ensemble / Bagging |
| Gradient Boosting | Boosting |
| XGBoost | Gradient boosting |
| CatBoost | Gradient boosting |
| Stacking Regressor | Stacking |
| Hybrid Voting Regressor | Voting |

A Dummy Regressor is also included as a baseline to assess the improvement achieved by the trained models over a simple prediction strategy.

## 5. Model Evaluation

Model performance is assessed using three regression evaluation metrics:

- **Mean Absolute Error (MAE):** Measures the average absolute difference between predicted and actual prices.
- **Root Mean Squared Error (RMSE):** Penalises larger prediction errors more heavily.
- **Coefficient of Determination (R²):** Measures the proportion of target variance explained by the model.

### Comparative results

The following results were obtained using the held-out test dataset.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Dummy Baseline | 28.2047 | 43.7291 | -0.0655 |
| Decision Tree | 6.6449 | 16.1148 | 0.8553 |
| Random Forest | 5.4081 | 13.6657 | 0.8959 |
| Extra Trees | **5.3031** | 14.3124 | 0.8859 |
| Gradient Boosting | 8.5252 | 18.8067 | 0.8029 |
| XGBoost | 5.9785 | 13.8999 | 0.8923 |
| CatBoost | 7.2337 | 14.9874 | 0.8748 |
| Stacking Regressor | 5.3382 | **13.6410** | **0.8963** |
| Hybrid Voting | 6.2007 | 14.3352 | 0.8855 |

### Key observations

- The **Stacking Regressor** achieved the lowest RMSE and highest R² among the evaluated models.
- **Extra Trees** achieved the lowest MAE.
- Random Forest demonstrated competitive performance across all three evaluation metrics.
- Boosting approaches produced different results, illustrating that increased model complexity does not automatically guarantee better predictive performance.
- The Hybrid Voting Regressor did not outperform the strongest individual ensemble approaches in this experiment.

The results demonstrate that model selection should consider multiple evaluation metrics rather than relying on a single performance measure.

## 6. Model Explainability

Predictive accuracy alone does not provide sufficient insight into how a machine learning model uses its input features.

To investigate model behaviour, **Permutation Feature Importance** was applied to the Stacking Regressor.

The analysis evaluates the reduction in predictive performance when individual features are randomly permuted.

### Most influential features

| Feature | Mean importance |
|---|---:|
| Brand | 10.5772 |
| YOM | 7.8477 |
| Engine (cc) | 6.7191 |
| Gear | 5.2270 |
| Millage (KM) | 4.9952 |
| Model | 2.6277 |
| Vehicle Age | 2.4206 |

Brand, year of manufacture, engine capacity, transmission type, and mileage emerged as particularly influential features.

These findings provide an indication of which vehicle characteristics the trained model relies on most when generating predictions. Feature importance indicates predictive contribution, not a causal relationship.

## 7. Interactive Dashboard

The trained machine learning pipeline is deployed through an interactive Streamlit application.

<details>
<summary><strong>Dashboard capabilities</strong></summary>

- Vehicle price prediction using the trained machine learning pipeline.
- Model performance comparison.
- Visualisation of regression evaluation metrics.
- Feature importance analysis.
- Interactive exploration of vehicle attributes.

</details>

**Access the deployed application:**

https://ensemblex-lk.streamlit.app/

The dashboard demonstrates how the trained model can be used to generate estimated asking prices from user-provided vehicle specifications.

## 8. Repository Structure

```text
EnsembleX-LK/
│
├── app.py
├── requirements.txt
│
├── ensemblex_lk_best_pipeline.joblib
│
├── model_comparison.csv
├── permutation_importance.csv
├── model_metadata.json
├── test_predictions.csv
│
└── README.md
```

| File | Description |
|---|---|
| `app.py` | Streamlit dashboard implementation |
| `requirements.txt` | Python dependencies |
| `ensemblex_lk_best_pipeline.joblib` | Saved trained preprocessing and prediction pipeline |
| `model_comparison.csv` | Comparative model evaluation results |
| `permutation_importance.csv` | Feature importance results |
| `model_metadata.json` | Supporting model metadata |
| `test_predictions.csv` | Actual and predicted test-set values |
| `README.md` | Project documentation |

## 9. Installation and Execution

### Prerequisites

- Python 3.11
- pip package manager

### Step 1: Clone the repository

```bash
git clone https://github.com/praveesha/EnsembleX-LK.git
cd EnsembleX-LK
```

### Step 2: Create a virtual environment

```bash
python -m venv venv
```

Activate the environment:

**Windows**
```bash
venv\Scripts\activate
```

**macOS / Linux**
```bash
source venv/bin/activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Launch the dashboard

```bash
streamlit run app.py
```

The application will be accessible through the local URL displayed in the terminal.

## 10. Limitations and Future Improvements

Although the framework demonstrates promising predictive performance, several limitations should be considered.

**Dataset limitations**
- The dataset represents a particular collection of Sri Lankan vehicle listings and may not capture the entire automotive market.
- Asking prices may differ from actual negotiated transaction prices.
- Market conditions and vehicle valuations can change over time.

**Modelling limitations**
- Performance is evaluated using a held-out test dataset.
- A single train-test split may not capture all sources of variation.
- Relationships between features such as year of manufacture, vehicle age, and mileage require careful interpretation.
- Predictions for vehicles with characteristics outside the training data distribution may be less reliable.

**Potential future improvements**
- Introduce cross-validation and systematic hyperparameter optimisation.
- Incorporate more recent and comprehensive Sri Lankan vehicle market data.
- Investigate temporal validation to assess performance on future listings.
- Explore additional explainability techniques.
- Evaluate model robustness across different vehicle categories and market segments.

## 11. Conclusion

EnsembleX LK demonstrates the application of multiple ensemble learning strategies to used vehicle price prediction in the Sri Lankan context.

Through comparative evaluation, the project investigates the strengths and limitations of bagging, boosting, voting, and stacking approaches. The results highlight the competitive performance of tree-based ensemble methods, with the Stacking Regressor achieving the strongest overall RMSE and R² performance in the evaluated experiment.

The integration of permutation feature importance provides additional insight into the predictive behaviour of the selected model, while the Streamlit dashboard translates the experimental work into an accessible interactive application.

Overall, the project brings together predictive modelling, comparative experimentation, explainability, and practical deployment within a single machine learning workflow.

---

## Technologies Used

- **Programming Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Visualisation:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Ensemble Learning:** Random Forest, Extra Trees, Gradient Boosting, XGBoost, CatBoost, Stacking and Voting
- **Explainability:** Permutation Feature Importance
- **Application Development:** Streamlit
- **Deployment:** Streamlit Community Cloud
- **Dataset:** Kaggle

---

<p align="center">
  <strong>EnsembleX LK</strong><br/>
  Explainable Machine Learning for Sri Lankan Vehicle Price Prediction
</p>