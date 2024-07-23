# Neo4j_Graph_based_Chatbot_on_pdf - Information Exactractor

Information Extractor is a Streamlit application that leverages NLP and machine learning techniques to extract relevant information from a knowledge base. The application uses spaCy for natural language processing, a Neo4j graph database for storing topics, and Sentence-Transformers for calculating similarity scores. It also includes a paraphrasing module to enhance query matching.

## Features

- Load and process a CSV file with sales data.
- Extract the month from a user-typed query using regular expressions.
- Filter and visualize sales data by month.
- Display key performance indicators (KPIs) for transactions, products, and total sales.
- Use spaCy, Sentence-Transformers, and TF-IDF for query similarity matching.
- Paraphrase queries using the Parrot model.
- Interactive Streamlit interface for querying and displaying information.

## Requirements

- Python 3.6 or higher
- Streamlit
- Pandas
- Matplotlib
- spacy
- py2neo
- scikit-learn
- scipy
- sentence-transformers
- parrot
- pyinstrument

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/Cilvo/Neo4j_Graph_based_Chatbot_on_pdf.git
    cd information-extractor
    ```

2. Install the required packages:

    ```bash
    pip install streamlit pandas matplotlib spacy py2neo scikit-learn scipy sentence-transformers parrot pyinstrument
    ```

3. Download the spaCy model:

    ```bash
    python -m spacy download en_core_web_md
    ```

## Usage

1. Run the Streamlit application:

    ```bash
    streamlit run app.py
    ```

2. The application will start in your default web browser. You can then type a query into the input box to extract relevant information from the knowledge base.

