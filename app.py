import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from pandasai import SmartDataframe
from langchain_groq.chat_models import ChatGroq

# Load environment variables (if you have .env file for API keys)
load_dotenv()

# Set up the ChatGroq LLM
llm = ChatGroq(
    model_name='llama3-70b-8192',
    api_key='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'  # Replace with your actual API key
)

# Streamlit App
st.title('Data Analysis and Insights with SmartDataframe')

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    try:
        # Load the data into a DataFrame
        data = pd.read_csv(uploaded_file)

        # Display the data
        st.write("## Uploaded Data")
        st.dataframe(data)

        # Create a SmartDataframe with the LLM
        df = SmartDataframe(data, config={'llm': llm})

        # Text input for user questions
        question = st.text_input("Ask a question about the data:")

        if question:
            # Generate insights based on user question
            response = df.chat(question)
            st.write("## Detailed Insights")
            st.write(response)

        # Button to describe the dataset
        if st.button("Describe the Dataset"):
            description = df.chat("Describe the dataset in a sentence , about what the data is? , what the datset tells about.")
            st.write("## Dataset Description")
            st.write(description)

        if st.button("About the dataset"):
            ans = df.chat("how and what ways this impact to stakeholders ?")
            st.write("## wait! Here are more insights")
            st.write(ans)

        # Data visualization options
        st.sidebar.title("Visualizations")
        visualization_type = st.sidebar.selectbox("Select Visualization Type", ["None", "Univariate", "Bivariate"])

        if visualization_type == "Univariate":
            x_axis = st.sidebar.selectbox("Select Axis", data.columns)
            if st.sidebar.button("Bar Chart"):
                fig, ax = plt.subplots(figsize=(10, 6))
                data[x_axis].value_counts().plot(kind='bar', ax=ax)
                ax.set_xlabel(x_axis)
                ax.set_ylabel('Frequency')
                ax.set_title(f'Bar Chart of {x_axis}')
                st.pyplot(fig)
            if st.sidebar.button("Line Chart"):
                fig, ax = plt.subplots(figsize=(10, 6))
                data[x_axis].plot(kind='line', ax=ax)
                ax.set_xlabel(x_axis)
                ax.set_ylabel('Values')
                ax.set_title(f'Line Chart of {x_axis}')
                st.pyplot(fig)
            if st.sidebar.button("Histogram"):
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(data[x_axis], bins=20)
                ax.set_xlabel(x_axis)
                ax.set_ylabel('Frequency')
                ax.set_title(f'Histogram of {x_axis}')
                st.pyplot(fig)

        elif visualization_type == "Bivariate":
            x_axis = st.sidebar.selectbox("X-axis", data.columns)
            y_axis = st.sidebar.selectbox("Y-axis", data.columns)
            if st.sidebar.button("Bar Chart"):
                fig, ax = plt.subplots(figsize=(12, 8))
                data[[x_axis, y_axis]].set_index(x_axis).plot(kind='bar', ax=ax)
                ax.set_xlabel(x_axis)
                ax.set_ylabel(y_axis)
                ax.set_title(f'Bar Chart of {y_axis} by {x_axis}')
                st.pyplot(fig)
            if st.sidebar.button("Line Chart"):
                fig, ax = plt.subplots(figsize=(12, 8))
                data[[x_axis, y_axis]].set_index(x_axis).plot(kind='line', ax=ax)
                ax.set_xlabel(x_axis)
                ax.set_ylabel(y_axis)
                ax.set_title(f'Line Chart of {y_axis} by {x_axis}')
                st.pyplot(fig)
            if st.sidebar.button("Scatter Plot"):
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.scatter(data[x_axis], data[y_axis])
                ax.set_xlabel(x_axis)
                ax.set_ylabel(y_axis)
                ax.set_title(f'Scatter Plot of {y_axis} vs {x_axis}')
                st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.write("Please upload a CSV file to get started.")
