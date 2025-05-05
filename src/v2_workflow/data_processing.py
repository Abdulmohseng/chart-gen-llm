import pandas as pd
from state import State
import streamlit as st
from langgraph.types import interrupt
from typing import Literal, Optional

def input_dataset(state: State):
    print("---input dataset---")
    prompt_message = """
📊 ENTER THE PATH TO YOUR DATASET:
- Provide the full path to your CSV file
- Example: data/sales_data.csv or /home/user/data/financial_data.csv
- The app will use this data to generate visualizations
"""
    value = interrupt(prompt_message)
    return {"file_path": value}

def summarize(state: State):
    print("---Step 1: Summarize---")
    try:
        df =  st.session_state.df # pd.read_csv(state['file_path'])
    except Exception as e:
        error_message = f"Failed to read dataset: {e}\nPlease check that the file exists and is in CSV format."
        print(error_message)
        return {'summary': [], 'is_applicable': False}

    summary = []
    summary.append(f"📊 DATASET OVERVIEW: {df.shape[0]} rows × {df.shape[1]} columns\n")
    summary.append("📋 COLUMN ANALYSIS:\n")

    for col in df.columns:
        dtype = df[col].dtype
        missing = df[col].isna().sum()
        non_missing = df.shape[0] - missing
        summary.append(f"📌 {col} ({dtype})")

        if dtype == 'object':
            unique = df[col].nunique()
            top = df[col].value_counts().idxmax() if unique > 0 else "N/A"
            freq = df[col].value_counts().max() if unique > 0 else 0
            summary.append(f"   • Unique values: {unique}, Most frequent: '{top}' ({freq} occurrences)")
            sample_values = df[col].dropna().unique()[:3]
            summary.append(f"   • Example values: {', '.join(map(str, sample_values))}")
        
        elif pd.api.types.is_numeric_dtype(df[col]):
            mean = df[col].mean()
            std = df[col].std()
            min_val = df[col].min()
            max_val = df[col].max()
            skew = df[col].skew()
            summary.append(f"   • Mean: {mean:.2f}, Std: {std:.2f}, Range: [{min_val:.2f} to {max_val:.2f}]")
            summary.append(f"   • Skewness: {skew:.2f} {'(Highly skewed)' if abs(skew) > 1 else '(Relatively symmetric)'}")

        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            summary.append(f"   • Time range: {df[col].min()} to {df[col].max()}")
            summary.append(f"   • Time span: {(df[col].max() - df[col].min()).days} days")

        summary.append(f"   • Missing: {missing} ({missing / df.shape[0] * 100:.1f}%)\n")

    # Add correlation analysis for numerical columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) >= 2:
        summary.append("📊 CORRELATION ANALYSIS:")
        corr_matrix = df[numeric_cols].corr()
        
        # Find strongest correlations
        corrs = []
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                col1, col2 = numeric_cols[i], numeric_cols[j]
                corr_val = corr_matrix.loc[col1, col2]
                if not pd.isna(corr_val):
                    corrs.append((col1, col2, corr_val))
        
        # Sort by absolute correlation value
        corrs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        # Report top 3 correlations
        for col1, col2, corr_val in corrs[:3]:
            strength = "Strong" if abs(corr_val) > 0.7 else "Moderate" if abs(corr_val) > 0.3 else "Weak"
            direction = "positive" if corr_val > 0 else "negative"
            summary.append(f"   • {col1} vs {col2}: {corr_val:.2f} ({strength} {direction} correlation)")

    return {
        'summary': "\n".join(summary),
        'is_applicable': True
    }

def validate_chart_code(state) -> Optional[Literal['user_change_request', 'generate_chart_code', None]]:
    """
    Step 5:
    Validate the chart code then either passes or fails.
    if pass --> user change requests
    if fail --> go to step 4 and generate the code again given the error message
    """
    print("---Step 5: chart validate---")
    if state['code_retry'] > 2:
        print("Enetered None")
        return None
    pass