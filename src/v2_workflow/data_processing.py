import pandas as pd
from state import State
import streamlit as st
from langgraph.types import interrupt

def input_dataset(state: State):
    print("---input dataset---")
    # file_path = input("Enter the path to your CSV dataset: (default data/japanvchina.csv)") or "data/japanvchina.csv"
    value = interrupt("What is the datatsets path?")
    return {"file_path": value}

def summarize(state: State):
    print("---Step 1: Summarize---")
    try:
        df = pd.read_csv(state['file_path'])
    except Exception as e:
        print(f"Failed to read dataset, try again: {e}")
        return {'summary': [], 'is_applicable': False}

    summary = []
    summary.append(f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns.\n")
    summary.append("Column Overview:\n")

    for col in df.columns:
        dtype = df[col].dtype
        missing = df[col].isna().sum()
        non_missing = df.shape[0] - missing
        summary.append(f"** {col} ({dtype})")

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
            summary.append(f"   • Mean: {mean:.2f}, Std: {std:.2f}, Min: {min_val}, Max: {max_val}")
            summary.append(f"   • Skewness: {skew:.2f}")

        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            summary.append(f"   • Range: {df[col].min()} to {df[col].max()}")

        summary.append(f"   • Missing: {missing} ({missing / df.shape[0] * 100:.1f}%)\n")

    return {
        'summary': "\n".join(summary),
        'is_applicable': True
    }

def validate_chart_code(state):
    """
    Step 5:
    Validate the chart code then either passes or fails.
    if pass --> user change requests
    if fail --> go to step 4 and generate the code again given the error message
    """
    print("---Step 5: chart validate---")
    pass