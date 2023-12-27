from io import BytesIO

import google.generativeai as palm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from langchain.agents.agent_types import AgentType
from langchain.llms import GooglePalm
from langchain_experimental.agents.agent_toolkits import create_csv_agent

# Change log
# -) delete unused generate_summary function
# -) delete unused nltk library
# -) delete unused dotenv library
# -) delete unused Image library from PIL
# -) delete unused sumy nlp, parserers, summarizers library
# -) add langchain section as third analysis type (generate_response funciton, import langchain lib., and creating the ui)


API_KEY = "AIzaSyD7eSg8_WIkx-URrLO0L9Fu6LYCL63Qh0Y"
palm.configure(api_key=API_KEY)


def generate_graphic_statistic(data):
    st.subheader("Graphic Analysis:")

    numeric_cols = data.select_dtypes(include="number").columns
    for col in numeric_cols:
        # Create the figure and axes correctly
        fig, ax = plt.subplots(figsize=(8, 6))

        # Create the histogram using Seaborn
        sns.histplot(data[col], bins=30, kde=True, color="orange", ax=ax)

        # Add labels and title
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")

        # Display the plot using st.pyplot
        st.pyplot(fig)

        st.write(
            f"The above plot shows the distribution of {col}. Each bar represents a bin, and the number on top of each bar indicates the frequency of data points in that bin."
        )

        # Additional information about the analysis
        st.write(f"Analysis Summary for {col}:")
        st.write(f" - Mean: {data[col].mean()}")
        st.write(f" - Median: {data[col].median()}")
        st.write(f" - Standard Deviation: {data[col].std()}")

        # Save plot to BytesIO and provide a download button
        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        st.download_button(
            label=f"Download {col} Distribution Plot",
            data=buffer,
            file_name=f"{col}_distribution_plot.png",
            key=f"{col}_distribution_plot",
        )


def generate_descriptive_statistic(data):
    st.subheader("Descriptive Statistics:")

    stats = data.describe()

    # Additional statistics
    stats.loc["range"] = stats.loc["max"] - stats.loc["min"]
    stats.loc["skew"] = data.skew()
    stats.loc["kurt"] = data.kurtosis()

    # Display the enhanced statistics
    st.dataframe(stats)

    # Additional statistics section
    st.subheader("Additional Statistics:")
    st.write(f" - Range: {stats.loc['range'].values}")
    st.write(f" - Skewness: {stats.loc['skew'].values}")
    st.write(f" - Kurtosis: {stats.loc['kurt'].values}")

    # Auto-generated conclusion
    conclusion = generate_conclusion(stats)
    st.subheader("Auto-Generated Conclusion:")
    st.write(conclusion)

    # Save statistics to CSV and provide a download button
    csv_data = stats.to_csv(index=True).encode()
    st.download_button(
        label="Download Descriptive Statistics",
        data=csv_data,
        file_name="descriptive_statistics.csv",
        key="descriptive_statistics",
    )


def generate_conclusion(stats):
    conclusion = (
        "Based on the descriptive statistics, we can make the following observations:\n"
    )

    # Example conclusions (you can customize these based on your dataset and analysis)
    for col in stats.columns:
        if col != "count":
            conclusion += f" - The {col} variable has a {get_tendency(stats.loc['mean'][col])} tendency.\n"

    return conclusion


def get_tendency(value):
    if value > 0:
        return "positive"
    elif value < 0:
        return "negative"
    else:
        return "neutral"


# Langchain Section
def generate_response(question, csv_file):
    # Generates a response to a question using the CSV agent.
    agent = create_csv_agent(
        GooglePalm(temperature=0.5, google_api_key=API_KEY),
        csv_file,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )
    # Configure the prompt
    response = agent.run("make it long answer with explanation: " + question)
    st.write("Nice question...")
    st.info(response)
    st.success("Analysis completed.")


def main():
    st.title("Dataset Analysis Tool")

    # Form to upload the CSV file
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(uploaded_file)
        # Display the dataset
        st.dataframe(df)

        # Select analysis type
        analysis_type = st.selectbox(
            "Select Analysis Type",
            ["graphic statistic", "descriptive statistic", "analyst assistance"],
        )

        # Add langchain section for analyst assistance
        if analysis_type == "analyst assistance":
            # Reset the file pointer to the beginning for re-use
            uploaded_file.seek(0)
            st.write("Hello, im your assistance")
            question = st.text_input("Can I help you?")
            if st.button("Generate Analysis"):
                generate_response(question, uploaded_file)
        else:
            # Button to generate analysis
            if st.button("Generate Analysis"):
                # Perform the selected analysis
                if analysis_type == "graphic statistic":
                    generate_graphic_statistic(df)
                elif analysis_type == "descriptive statistic":
                    generate_descriptive_statistic(df)
                st.success("Analysis completed.")


# Run the app
if __name__ == "__main__":
    main()
