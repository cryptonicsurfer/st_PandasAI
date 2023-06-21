from dotenv import load_dotenv
import os
import streamlit as st
import pandas as pd
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
import requests
import json
import numpy as np

load_dotenv()
API_KEY = os.environ['OPENAI_API_KEY']

llm = OpenAI(api_token=API_KEY)
pandas_ai = PandasAI(llm)

st.set_page_config(layout="wide")

with st.sidebar:
    st.image('image3.jpg')
    st.title("Ställ frågor i ren text till din data")

with open('directus_api.txt', 'r') as f:
    lines = f.readlines()

variables = {}

for line in lines:
    name, value = line.split('=', 1)
    name = name.strip()
    value = value.strip().replace('"', '')
    variables[name] = value

col1, col2 = st.columns(2)

with col1:
    variable_name = st.selectbox("Choose a category", list(variables.keys()))
    url = variables[variable_name]
    response = requests.get(url)
    data = response.json()
    data = data['data']

    df = pd.DataFrame(data)
    df.columns = df.columns.str.lower()
    if 'datum' in df.columns:
        df['datum'] = pd.to_datetime(df['datum'])

    st.write(df.head(4))

    all_columns = df.columns.tolist()
    all_columns.remove('datum')

    all_columns = ['None'] + all_columns

    category_column = st.selectbox("Select the category column(s) if data is in long format", all_columns)
    value_column = st.selectbox("Select the value column(s)", all_columns)

    df_final = pd.DataFrame()

    if category_column != 'None':  # if a category column is selected, treat data as long format
        df_pivot = df.pivot(index='datum', columns=category_column, values=value_column)
        st.line_chart(df_pivot)
        df_final = df_pivot
      
    else:  # if no category column is selected or 'None' is selected, treat data as wide format
        if value_column != 'None':  # ensure that value_column is not 'None'
            df_melt = pd.melt(df, id_vars=['datum'], value_vars=value_column)
            df_melt.set_index('datum', inplace=True)
            st.line_chart(df_melt)
            df_final = df_melt

with col2:
    prompt = st.text_area("skriv din fråga")

    if st.button('generera AI fråga och svar'):
        if prompt:
            st.write("PandasAI jobbar på att generera svar, vänta...")
            st.write(pandas_ai.run(df_final, prompt=prompt))
        else:
            st.warning("skriv in din fråga")
    st.write(df_final, (10))
