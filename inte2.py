import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

import argparse
from dataclasses import dataclass
from translate import Translator
import streamlit as st

from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Add radio buttons for gender selection
gender = st.radio("Select gender:", ("Male", "Female"))

# Set the sheet name based on the selected gender
sheet_name = "male" if gender == "Male" else "female"

# Read the Excel file
df = pd.read_excel('BPmale.xlsx', sheet_name=sheet_name)

# Extract the data for age, SBP, and DBP
age = df['age']
sbp = df['SBP']
dbp = df['DBP']

# Create a sidebar for user input
st.sidebar.title('User Input')
user_age = st.sidebar.number_input('Enter age:', value=30)
user_sbp = st.sidebar.number_input('Enter SBP value:', value=120.0)
user_dbp = st.sidebar.number_input('Enter DBP value:', value=80.0)

# Create a figure and axis
fig, ax = plt.subplots()

# Plot the data
ax.plot(age, sbp, marker='o', linestyle='-', color='blue', label='SBP')
ax.plot(age, dbp, marker='o', linestyle='-', color='red', label='DBP')

# Fill the area between the lines
ax.fill_between(age, sbp, dbp, where=(sbp > dbp), facecolor='lightblue', alpha=0.5)
ax.fill_between(age, sbp, dbp, where=(dbp > sbp), facecolor='lightpink', alpha=0.5)

# Plot user input as red stars
ax.plot(user_age, user_sbp, marker='^', markersize=10, color='black', label='User SBP')
ax.plot(user_age, user_dbp, marker='v', markersize=10, color='black', label='User DBP')

# Customize the plot
ax.set_xlabel('Age')
ax.set_ylabel('Blood Pressure')
ax.set_title('Blood Pressure Threshold vs. Age')
ax.legend()

# Display the plot using Streamlit
st.pyplot(fig)

# Optionally, you can display the data as a table
st.dataframe(df)

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    st.title("ความเสี่ยงต่อโรค")
    user_blood_pressure = st.text_input("กรุณาลงข้อมูลความดันโลหิต", value=f"BP {user_sbp}/{user_dbp}")

    if user_blood_pressure:
        # Google Translate
        try:
            translator = Translator(from_lang='th', to_lang='en')
            translated_text = translator.translate(user_blood_pressure)
        except Exception as e:
            st.write(f"Error translating: {str(e)}")
            return

        # Rest of the code...
        st.write(f"Translated Text: {translated_text}") 

        # Prepare the DB.
        openai_api_key = "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"  # Replace with your actual OpenAI API key
        if not openai_api_key:
            st.write("OpenAI API key is not provided.")
            return

        embedding_function = OpenAIEmbeddings(openai_api_key=openai_api_key)
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

        # Search the DB.
        results = db.similarity_search_with_relevance_scores(translated_text, k=3)
        if not results or (results and results[0][1] < 0.7):
            st.write("ไม่สามารถค้นหาคำตอบขณะนี้ได้ โปรดติดต่อแพทย์ของท่าน")
            return

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=translated_text)
        st.write(prompt)

        model = ChatOpenAI()
        response_text = model.predict(prompt)

        sources = [doc.metadata.get("source", None) for doc, _score in results]
        formatted_response = f"<span style='color:red'>{response_text}</span>\nSources: {sources}"
        st.write(formatted_response, unsafe_allow_html=True)

        # Choose the image based on blood pressure values
        if user_sbp <= 140 and user_dbp <= 90:
            st.image("image1.png", caption="Case: SBP <= 140 and DBP <= 90")
        elif user_sbp >= 140 and user_sbp <= 180:
            st.image("image2.png", caption="Case: SBP >= 140 and <= 180")
        elif user_sbp >= 180:
            st.image("image3.png", caption="Case: SBP >= 180")

if __name__ == "__main__":
    main()