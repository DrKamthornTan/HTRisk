import argparse
from dataclasses import dataclass

from translate import Translator
import streamlit as st

from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    st.title("ความเสี่ยงต่อโรค")
    query_text = st.text_input("กรุณาลงข้อมูล")

    if query_text:
        # Google Translate
        try:
            translator = Translator(from_lang='th', to_lang='en')
            translated_text = translator.translate(query_text)
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

if __name__ == "__main__":
    main()
