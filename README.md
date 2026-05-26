# Credit Risk Modeling Streamlit App

## Overview
This project evaluates credit risk using a finalized logistic regression model and presents the result in a Streamlit app. The repository also includes model-development experimentation in `credit_risk_model_codebasics.ipynb` where Logistic Regression, Random Forest, and XGBoost were explored, but the production app uses the final Logistic Regression model.

## What was improved
- Explainability: feature importance and impact are shown in the UI.
- Risk recommendations: risk category, approval recommendation, interest range, and suggested loan amount.
- Business summary: a human-readable narrative explanation of the prediction.
- UI upgrades: risk meter, charts, and downloadable assessment report.
- Optional GenAI layer: when `GROQ_API_KEY` and `GROQ_MODEL` are configured, the app can generate a richer natural-language summary from Groq.

## Project structure
- `app/main.py` — main Streamlit user interface
- `app/prediction_helper.py` — prediction, scoring, explainability and recommendation logic
- `artifacts/model_data.joblib` — deployed logistic regression model, scaler, and feature metadata
- `credit_risk_model_codebasics.ipynb` — notebook containing exploratory model experiments and evaluation
- `dataset/` — source CSV datasets used for database creation
- `create_db.py` — loads dataset CSVs into `credit_risk.db`

## Run the app
1. Activate the Python environment.
2. From `Project2_StreamlitApp_Resources/`, run:
```bash
streamlit run app/main.py
```

## Optional Groq integration
To enable the GenAI summary, set the environment variables:
```bash
set GROQ_API_KEY=your_api_key
set GROQ_MODEL=your_model_name
```

## Final model and scoring logic
- The deployed app uses the final **Logistic Regression** model stored in `artifacts/model_data.joblib`.
- Input features are scaled using the stored `MinMaxScaler` before prediction.
- The model computes a weighted linear combination of features with learned coefficients, adds the intercept, then applies the logistic (sigmoid) function to produce a default probability.
- The credit score is derived from the non-default probability and scaled to a 300–900 range.
- Feature contributions shown in the UI are calculated as `feature_value × coefficient`, so positive contributions increase default risk and negative contributions reduce it.
- The risk category, approval recommendation, interest range, and suggested loan amount are determined by deterministic thresholds on the default probability and derived credit score.

## Future improvements
- Add real SHAP/LIME explanations for stronger interpretability.
- Expand the model comparison section to compare baseline models.
- Add a dedicated dashboard for portfolio-level risk analysis.
- Improve the dataset pipeline with data cleaning, feature engineering, and hyperparameter tuning.
