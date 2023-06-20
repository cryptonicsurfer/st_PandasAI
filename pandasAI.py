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
    st.image('image1.jpg')
    st.title("Ställ frågor i ren text till din data")


# Load the text file and parse it
with open('directus_api.txt', 'r') as f:
    lines = f.readlines()

# Initialize an empty dictionary
variables = {}

# Parse each line in the text file
for line in lines:
    # Split the line into a name and a value
    name, value = line.split('=')
    
    # Remove unnecessary characters like quotation marks and newline characters
    name = name.strip()
    value = value.strip().replace('"', '')
    
    # Add the name and value to the dictionary
    variables[name] = value


col1, col2 = st.columns(2)

with col1:
    # Create a dropdown menu in Streamlit with the keys of the dictionary
    variable_name = st.selectbox("Choose a category", list(variables.keys()))

    # Get the URL for the selected variable
    url = variables[variable_name]

    # Make a GET request to the API endpoint
    response = requests.get(url)

    #convert json response to python object
    data = response.json()

    #extract the "data" field from the response to an array
    data=data['data']

    # Convert the list of dictionaries to a Pandas DataFrame
    df = pd.DataFrame(data)

    # Create a multi-select box for selecting columns and convert dict to df
    selected_columns = st.multiselect("Select columns to keep", df.columns.tolist())
    df = df[selected_columns]

    if 'datum' in df.columns:
        df['datum'] = pd.to_datetime(df['datum'])
        df.set_index('datum', inplace=True)

    # Display the DataFrame in Streamlit
    st.write(df.head(4))

# Create a multi-select box for selecting the category column
category_column = st.selectbox("Select the category column", df.columns.tolist())

# Create a multi-select box for selecting the value column
value_column = st.selectbox("Select the value column", df.columns.tolist())

# Pivot your DataFrame if the selected category and value columns are in the DataFrame
if {category_column, value_column}.issubset(df.columns):
    df_pivot = df.pivot(index='datum', columns=category_column, values=value_column)

    # Display the pivoted DataFrame in Streamlit
    st.write(df_pivot.head(4))

    # Create a multi-select box for selecting categories to display
    selected_categories = st.multiselect("Select categories to display", df[category_column].unique().tolist())

    # Filter the pivoted DataFrame to only include the selected categories
    df_pivot = df_pivot[selected_categories]

    # Draw a line chart for your pivoted DataFrame
    st.line_chart(df_pivot)
    


with col2:
    prompt = st.text_area("skriv din fråga")

    if st.button('generera AI fråga och svar'):
        if prompt:
            st.write("PandasAI jobbar på att generera svar, vänta...")
            st.write(pandas_ai.run(df, prompt=prompt))
        else:
            st.warning("skriv in din fråga")


