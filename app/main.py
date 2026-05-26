import streamlit as st
import pandas as pd
from prediction_helper import predict

st.set_page_config(
    page_title='AI-Assisted Credit Risk Assessment System',
    page_icon='📊',
    layout='wide'
)

st.title('AI-Assisted Credit Risk Assessment System')
st.markdown('Use the form below to estimate default risk, view feature impact, and receive actionable credit recommendations.')

with st.sidebar:
    st.header('Applicant Profile')
    age = st.number_input('Age', min_value=18, step=1, max_value=100, value=28)
    income = st.number_input('Income', min_value=0, value=1200000)
    loan_amount = st.number_input('Loan Amount', min_value=0, value=2560000)
    loan_tenure_months = st.number_input('Loan Tenure (months)', min_value=0, step=1, value=36)
    avg_dpd_per_delinquency = st.number_input('Avg DPD', min_value=0, value=20)
    delinquency_ratio = st.number_input('Delinquency Ratio', min_value=0, max_value=100, step=1, value=30)
    credit_utilization_ratio = st.number_input('Credit Utilization Ratio', min_value=0, max_value=100, step=1, value=30)
    num_open_accounts = st.number_input('Open Loan Accounts', min_value=1, max_value=10, step=1, value=2)
    residence_type = st.selectbox('Residence Type', ['Owned', 'Rented', 'Mortgage'])
    loan_purpose = st.selectbox('Loan Purpose', ['Education', 'Home', 'Auto', 'Personal'])
    loan_type = st.selectbox('Loan Type', ['Unsecured', 'Secured'])
    st.markdown('---')
    st.write('Loan-to-income ratio is calculated as loan amount divided by income.')

loan_to_income_ratio = loan_amount / income if income > 0 else 0

if st.button('Calculate Risk'):
    result = predict(
        age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
        delinquency_ratio, credit_utilization_ratio, num_open_accounts,
        residence_type, loan_purpose, loan_type
    )

    default_probability = result['default_probability']
    credit_score = result['credit_score']
    rating = result['rating']
    contributions = result['contributions']
    recommendation = result['recommendation']
    summary = result['summary']

    st.markdown('## Credit Risk Assessment')
    top_metrics, chart_col = st.columns([1.5, 1])

    with top_metrics:
        st.metric('Default Probability', f'{default_probability:.1%}', delta=None)
        st.metric('Estimated Credit Score', credit_score, delta=rating)
        st.write(f'**Risk category:** {recommendation["risk_category"]}')
        st.write(f'**Approval recommendation:** {recommendation["approval_recommendation"]}')
        st.write(f'**Suggested interest range:** {recommendation["interest_rate"]}')
        st.write(f'**Suggested maximum loan amount:** ₹{recommendation["max_loan_amount"]:,}')

    with chart_col:
        st.write('### Risk Meter')
        st.progress(min(max(default_probability, 0), 1))
        if contributions:
            feature_chart = pd.DataFrame(
                [{'Feature': item['label'], 'Impact': item['contribution']} for item in contributions[:6]]
            ).set_index('Feature')
            st.bar_chart(feature_chart)

    st.markdown('### AI-Assisted Summary')
    st.info(summary)

    with st.expander('Feature importance and explanations'):
        st.write('The model weights show which features are driving the prediction:')
        contribution_df = pd.DataFrame(contributions)
        contribution_df['impact_direction'] = contribution_df['contribution'].apply(
            lambda x: 'Higher risk' if x > 0 else 'Lower risk'
        )
        st.dataframe(contribution_df[['label', 'value', 'coefficient', 'contribution', 'impact_direction']])

    with st.expander('Actionable recommendations'):
        for item in recommendation['action_items']:
            st.write(f'- {item}')

    report_text = f"""
Credit Risk Report
===================
Applicant age: {age}
Income: {income}
Loan amount: {loan_amount}
Loan tenure: {loan_tenure_months} months
Average DPD: {avg_dpd_per_delinquency}
Delinquency ratio: {delinquency_ratio}%
Credit utilization ratio: {credit_utilization_ratio}%
Open loan accounts: {num_open_accounts}
Residence type: {residence_type}
Loan purpose: {loan_purpose}
Loan type: {loan_type}

Default probability: {default_probability:.1%}
Credit score: {credit_score}
Rating: {rating}
Risk category: {recommendation['risk_category']}
Approval recommendation: {recommendation['approval_recommendation']}
Suggested interest rate: {recommendation['interest_rate']}
Suggested max loan amount: ₹{recommendation['max_loan_amount']:,}

Summary:
{summary}

Recommendations:
"""
    for item in recommendation['action_items']:
        report_text += f'- {item}\n'

    st.download_button(
        label='Download assessment report',
        data=report_text,
        file_name='credit_risk_report.txt',
        mime='text/plain'
    )

# Footer
st.markdown('---')
st.caption('Enhanced credit risk assessment with explainability, recommendations, and an optional GenAI-driven summary.')
