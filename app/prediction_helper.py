import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv

# Load the .env file located next to this helper module
ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
load_dotenv(ENV_PATH)

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

GROQ_MODEL = os.getenv('GROQ_MODEL')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'artifacts', 'model_data.joblib')

model_data = joblib.load(MODEL_PATH)
model = model_data['model']
scaler = model_data['scaler']
features = list(model_data['features'])
cols_to_scale = list(model_data['cols_to_scale'])

FEATURE_LABELS = {
    'age': 'Applicant Age',
    'loan_tenure_months': 'Loan Tenure (months)',
    'number_of_open_accounts': 'Open Loan Accounts',
    'credit_utilization_ratio': 'Credit Utilization Ratio',
    'loan_to_income': 'Loan-to-Income Ratio',
    'delinquency_ratio': 'Delinquency Ratio',
    'avg_dpd_per_delinquency': 'Average DPD',
    'residence_type_Owned': 'Residence Owned',
    'residence_type_Rented': 'Residence Rented',
    'loan_purpose_Education': 'Loan Purpose: Education',
    'loan_purpose_Home': 'Loan Purpose: Home',
    'loan_purpose_Personal': 'Loan Purpose: Personal',
    'loan_type_Unsecured': 'Loan Type: Unsecured'
}


def label(feature_name):
    return FEATURE_LABELS.get(feature_name, feature_name.replace('_', ' ').title())


def prepare_input(age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
                    delinquency_ratio, credit_utilization_ratio, num_open_accounts, residence_type,
                    loan_purpose, loan_type):
    input_data = {
        'age': age,
        'loan_tenure_months': loan_tenure_months,
        'number_of_open_accounts': num_open_accounts,
        'credit_utilization_ratio': credit_utilization_ratio,
        'loan_to_income': loan_amount / income if income > 0 else 0,
        'delinquency_ratio': delinquency_ratio,
        'avg_dpd_per_delinquency': avg_dpd_per_delinquency,
        'residence_type_Owned': 1 if residence_type == 'Owned' else 0,
        'residence_type_Rented': 1 if residence_type == 'Rented' else 0,
        'loan_purpose_Education': 1 if loan_purpose == 'Education' else 0,
        'loan_purpose_Home': 1 if loan_purpose == 'Home' else 0,
        'loan_purpose_Personal': 1 if loan_purpose == 'Personal' else 0,
        'loan_type_Unsecured': 1 if loan_type == 'Unsecured' else 0,
        'number_of_dependants': 1,
        'years_at_current_address': 1,
        'zipcode': 1,
        'sanction_amount': 1,
        'processing_fee': 1,
        'gst': 1,
        'net_disbursement': 1,
        'principal_outstanding': 1,
        'bank_balance_at_application': 1,
        'number_of_closed_accounts': 1,
        'enquiry_count': 1
    }

    df = pd.DataFrame([input_data])
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    df = df[features]
    return df


def get_feature_contributions(input_df):
    coeffs = model.coef_.flatten()
    values = input_df.iloc[0].values
    contributions = values * coeffs

    feature_contributions = []
    for feature, coeff, value, contribution in zip(features, coeffs, values, contributions):
        feature_contributions.append({
            'feature': feature,
            'label': label(feature),
            'value': float(value),
            'coefficient': float(coeff),
            'contribution': float(contribution)
        })

    return sorted(feature_contributions, key=lambda x: abs(x['contribution']), reverse=True)


def get_risk_profile(default_probability, credit_score):
    if default_probability >= 0.65:
        return {
            'risk_category': 'High risk',
            'approval': 'Decline or require stronger collateral',
            'interest_rate': '18-24%',
            'score_label': 'Poor'
        }
    if default_probability >= 0.40:
        return {
            'risk_category': 'Moderate risk',
            'approval': 'Approve with conditions',
            'interest_rate': '12-18%',
            'score_label': 'Average'
        }
    return {
        'risk_category': 'Low risk',
        'approval': 'Approve at competitive rate',
        'interest_rate': '6-12%',
        'score_label': 'Good'
    }


def get_actionable_recommendations(default_probability, credit_score, income, credit_utilization_ratio,
                                   delinquency_ratio, avg_dpd_per_delinquency, loan_to_income):
    profile = get_risk_profile(default_probability, credit_score)

    if profile['risk_category'] == 'High risk':
        max_loan = 0
        advisory = [
            'Reduce credit utilization below 30%.',
            'Lower delinquency ratio and resolve overdue payments.',
            'Limit active loan accounts until payment behavior improves.'
        ]
    elif profile['risk_category'] == 'Moderate risk':
        max_loan = min(int(income * 2), 5000000)
        advisory = [
            'Stay below a 30% credit utilization ratio.',
            'Pay down overdue installments to lower delinquency.',
            'Maintain stable income and avoid new credit inquiries.'
        ]
    else:
        max_loan = min(int(income * 4), 15000000)
        advisory = [
            'Keep credit utilization under 30%.',
            'Continue on-time payments to preserve your score.',
            'Avoid high-risk unsecured loans when possible.'
        ]

    if loan_to_income > 0.5:
        advisory.append('Reduce loan-to-income ratio by increasing income or lowering loan amount.')
    if credit_utilization_ratio > 30:
        advisory.append('Reduce credit utilization below 30% to improve eligibility.')
    if delinquency_ratio > 20 or avg_dpd_per_delinquency > 0:
        advisory.append('Address delinquent accounts quickly to rebuild credit trust.')

    return {
        'max_loan_amount': max_loan,
        'interest_rate': profile['interest_rate'],
        'action_items': advisory,
        'approval_recommendation': profile['approval'],
        'risk_category': profile['risk_category']
    }


def build_summary(default_probability, credit_score, rating, contributors):
    summary = [
        f'Applicant has a {rating.lower()} credit rating.',
        f'Estimated default probability is {default_probability:.1%}.',
        f'Credit score is {credit_score}.'
    ]

    if contributors:
        top = contributors[:3]
        summary.append('Key drivers:')
        for contributor in top:
            sign = 'increases' if contributor['contribution'] > 0 else 'reduces'
            summary.append(f"{contributor['label']} ({contributor['value']:.1f}) {sign} default risk.")

    return ' '.join(summary)


def generate_risk_summary(default_probability, credit_score, rating, contributions):
    if GROQ_AVAILABLE and GROQ_MODEL and GROQ_API_KEY:
        top_contributors = contributions[:3]
        prompt_lines = [
            'You are a financial risk analyst.',
            f'Applicant default probability: {default_probability:.2%}.',
            f'Credit score: {credit_score}.',
            f'Rating: {rating}.',
            'Use the top contributors to explain why the applicant is high or low risk.'
        ]
        for entry in top_contributors:
            prompt_lines.append(f"{entry['label']}: {entry['value']} contributes {entry['contribution']:.3f}.")
        prompt_lines.append('Provide a short, business-friendly risk summary and recommendation.')

        try:
            groq_client = Groq()
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{'role': 'user', 'content': '\n'.join(prompt_lines)}],
                max_tokens=220,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception:
            pass

    return build_summary(default_probability, credit_score, rating, contributions)


def predict(age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
            delinquency_ratio, credit_utilization_ratio, num_open_accounts,
            residence_type, loan_purpose, loan_type):
    input_df = prepare_input(age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
                             delinquency_ratio, credit_utilization_ratio, num_open_accounts, residence_type,
                             loan_purpose, loan_type)

    default_probability, credit_score, rating = calculate_credit_score(input_df)
    contributions = get_feature_contributions(input_df)
    recommendation = get_actionable_recommendations(default_probability, credit_score, income,
                                                   credit_utilization_ratio, delinquency_ratio,
                                                   avg_dpd_per_delinquency, input_df.iloc[0]['loan_to_income'])
    summary = generate_risk_summary(default_probability, credit_score, rating, contributions)

    return {
        'default_probability': default_probability,
        'credit_score': credit_score,
        'rating': rating,
        'contributions': contributions,
        'recommendation': recommendation,
        'summary': summary
    }


def calculate_credit_score(input_df, base_score=300, scale_length=600):
    # here we are calculating the log-odds (x) using the linear combination of features and model coefficients, 
    # then applying the logistic function to get the default probability. 
    # The credit score is derived from the non-default probability, scaled to fit within a range of 300 to 900.
    x = np.dot(input_df.values, model.coef_.T) + model.intercept_

    # Apply the logistic function to calculate the probability
    default_probability = 1 / (1 + np.exp(-x))

    non_default_probability = 1 - default_probability

    # Convert the probability to a credit score, scaled to fit within 300 to 900
    credit_score = base_score + non_default_probability.flatten() * scale_length

    # Determine the rating category based on the credit score
    def get_rating(score):
        if 300 <= score < 500:
            return 'Poor'
        elif 500 <= score < 650:
            return 'Average'
        elif 650 <= score < 750:
            return 'Good'
        elif 750 <= score <= 900:
            return 'Excellent'
        else:
            return 'Undefined'  # in case of any unexpected score

    rating = get_rating(credit_score[0])
    return default_probability.flatten()[0], int(credit_score[0]), rating