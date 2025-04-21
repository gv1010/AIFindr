import streamlit as st
import pandas as pd
from vector_db import VectorDB
from index_manager import IndexManager
from query_processor import QueryProcessor
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import logging

# Configure logging
logging.captureWarnings(True)

# Initialize components
@st.cache_resource
def initialize_components():
    vector_db = VectorDB(vectorstore_dir="./faiss_index")
    index_manager = IndexManager(docs_dir="./documents")
    index_manager.load_and_index_documents(vector_db)
    query_processor = QueryProcessor(vector_db=vector_db)
    return query_processor

# Initialize LLM chains
@st.cache_resource
def initialize_llm():
    llm = Ollama(model="qwen:0.5b")
    
    # Prompt for query enhancement with generalized persona
    enhance_prompt = PromptTemplate(
        input_variables=["query"],
        template="Enhance the following query by adding a single sentence that reflects a diverse, contextually relevant persona with varied interests and vibe, keeping it concise and natural: {query}"
    )
    enhance_chain = LLMChain(llm=llm, prompt=enhance_prompt)
    
    # Prompt for explanation
    explanation_prompt = PromptTemplate(
        input_variables=["query", "document"],
        template="Explain why the following document matches the query.\nQuery: {query}\nDocument: {document}\nExplanation:"
    )
    explanation_chain = LLMChain(llm=llm, prompt=explanation_prompt)
    
    return enhance_chain, explanation_chain

# Generate LLM explanation using Ollama
def get_llm_explanation(query, result_text, explanation_chain):
    """Generate an explanation using Ollama qwen:0.5b via LangChain."""
    try:
        response = explanation_chain.run(query=query, document=result_text[:1000])  # Limit document length
        return response.strip() or "No explanation provided."
    except Exception as e:
        return f"Error generating explanation: {str(e)}"

# Streamlit app
def main():
    st.title("Vector Search with LLM Explanation")

    # Initialize query processor and LLM
    query_processor = initialize_components()
    # explanation_chain = initialize_llm()
    enhance_chain, explanation_chain = initialize_llm()

    # Search bar
    query = st.text_input("Enter your search query:", placeholder="Type your query here...")

    # Search button
    if st.button("Search"):
        if query:
            with st.spinner("Searching..."):
                try:
                    enhanced_query = enhance_chain.run(query=query).strip()
                    if not enhanced_query:
                        enhanced_query = query  # Fallback to original if LLM fails
                except Exception as e:
                    st.error(f"Error enhancing query: {e}")
                    enhanced_query = query  # Fallback to original
                st.write(f"Enhanced Query: {enhanced_query}")
                
                # Get top-k results
                top_k_results = query_processor.get_top_k_results(enhanced_query, k=3)

                if top_k_results:
                    st.subheader("Search Results")
                    # Prepare data for display
                    results_df = pd.DataFrame([
                        {
                            "ID": i + 1,
                            "Text": result["text"],
                            "Metadata": result["metadata"],
                        }
                        for i, result in enumerate(top_k_results)
                    ])
                    st.dataframe(results_df[["ID", "Text"]], use_container_width=True)

                    # Display LLM explanations
                    st.subheader("LLM Explanations")
                    for i, result in enumerate(top_k_results):
                        explanation = get_llm_explanation(query, result["text"], explanation_chain)
                        with st.expander(f"Explanation for Result {i + 1}: {result['text'][:50]}..."):
                            st.write(explanation)
                else:
                    st.warning("No results found.")
        else:
            st.error("Please enter a query.")

if __name__ == "__main__":
    main()