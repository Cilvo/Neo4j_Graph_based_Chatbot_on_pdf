import spacy
import time
import pandas as pd
from py2neo import Graph
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance
from sentence_transformers import SentenceTransformer
from parrot import Parrot
import warnings
warnings.filterwarnings("ignore")
from pyinstrument import Profiler


class InformationExtractor:
    def __init__(self):
        # Load the spacy model
        self.nlp = spacy.load("en_core_web_md")

        # Create a Graph object and connect to the database
        self.graph = Graph("bolt://localhost:7689", auth=("neo4j", "password"))

        # Initialize Sentence-Transformers model
        self.sentrans = SentenceTransformer("distilbert-base-nli-mean-tokens")

        # Load topics and their corresponding paragraphs from the database
        self.topics, self.topics_paras = self.load_topics()

    def load_topics(self):
        # Define a Cypher query to get all nodes with the given label and property value
        query = "MATCH (n)-[r:Topic]->(m) RETURN r.data AS headline, r.heading_list as list"

        # Run the query and extract information
        results = self.graph.run(query)
        df = pd.DataFrame(results.data())
        topics = list(set(df.headline))
        topics_paras = list(df.list)

        return topics, topics_paras

    def paraphrase(self, query):
        """
        Paraphrases the given query using the Parrot paraphraser model.

        Args:
            query (str): The input query to paraphrase.

        Returns:
            list: A list of paraphrased versions of the query.
        """
        # Initialize Parrot paraphraser
        self.parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5", use_gpu=True)

        # Use the Parrot paraphraser to augment the input phrase (query)
        para_phrases = self.parrot.augment(input_phrase=query)

        # Initialize an empty list to store the paraphrased versions
        para_list = []

        # Check if any paraphrases are generated
        if para_phrases is not None:
            # Iterate through each paraphrase and append it to the para_list
            for para_phrase in para_phrases:
                para_list.append(para_phrase[0])

        # Return the list of paraphrased versions of the query
        return para_list


    def process(self, query):
        """
        Preprocesses the given query by performing tokenization, lowercasing, stopword removal,
        lemmatization, and punctuation removal.

        Args:
            query (str): The input query to preprocess.

        Returns:
            str: The processed version of the query.
        """
        # Tokenization
        doc_query = self.nlp(query)
        tokens = [token.text for token in doc_query]

        # Lowercasing
        lowercase_tokens = [token.lower() for token in tokens]

        # Stopword Removal
        filtered_tokens = [
            token for token in lowercase_tokens if not self.nlp.vocab[token].is_stop
        ]

        # Lemmatization
        lemmatized_tokens = [token.lemma_ for token in self.nlp(" ".join(filtered_tokens))]

        # Punctuation Removal
        punct_removed_tokens = [
            token for token in lemmatized_tokens if not self.nlp.vocab[token].is_punct
        ]

        # Join the tokens to form the processed query
        query_processed = " ".join(punct_removed_tokens)

        # Return the processed query
        return query_processed

    def spacy_sim(self, query, text):
        """
        Calculates the similarity score between the given query and text using spaCy's similarity measure.

        Args:
            query (str): The query string.
            text (str): The text string to compare with the query.

        Returns:
            float: The similarity score between the query and text.
        """
        doc1 = self.nlp(query)
        doc2 = self.nlp(text)
        sim = doc1.similarity(doc2)
        return sim
    
    def sent_trans_sim(self, query, text):
        """
        Calculates the similarity score between the given query and text using Sentence-Transformers.

        Args:
            query (str): The query string.
            text (str): The text string to compare with the query.

        Returns:
            float: The similarity score between the query and text.
        """
        sentences = [query, text]

        # Encode the sentences using Sentence-Transformers
        sentence_embeddings = self.sentrans.encode(sentences)

        # Calculate the cosine similarity between the sentence embeddings
        similarity_score = 1 - distance.cosine(sentence_embeddings[0], sentence_embeddings[1])

        return similarity_score
    


    def tf_idf_sim(self, query, text):
        """
        Calculates the similarity score between the given query and text using TF-IDF.

        Args:
            query (str): The query string.
            text (str): The text string to compare with the query.

        Returns:
            float: The similarity score between the query and text.
        """
        # Create TF-IDF vectors for the strings
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform([query, text])

        # Calculate cosine similarity between the vectors
        sim = cosine_similarity(tfidf[0], tfidf[1])[0][0]

        return sim

    def sim_list_maker(self, query_processed, sim_func):
        """
        Generates a list of similarity scores between the processed query and a list of topics using a specified similarity function.

        Args:
            query_processed (str): The processed query string.
            sim_func (function): The similarity function to calculate the similarity scores.

        Returns:
            list: A list of similarity scores between the query and topics.
        """
        sim_list = []
        for topic in self.topics:
            sim = sim_func(query_processed, topic)
            sim_list.append(sim)
        return sim_list

    def sim_list_maker_paraphrase(self, query_processed, sim_func):
        """
        Calculates the similarity scores between the paraphrased queries and the topics using a specified similarity function.

        Args:
            query_processed (str): The processed query string.
            sim_func (function): The similarity function to calculate the similarity scores.

        Returns:
            list: A list of similarity scores between the paraphrased queries and topics.
        """
        queries = self.paraphrase(query_processed)
        sim_list = []
        for topic_para in self.topics_paras:
            topic_sim = []
            for topic in topic_para:
                sim_query = []
                for query in queries:
                    sim = sim_func(query, topic)
                    sim_query.append(sim)
                avg_sim_query = sum(sim_query) / len(sim_query)
                topic_sim.append(avg_sim_query)
            avg_topic = sum(topic_sim) / len(topic_sim)
            sim_list.append(avg_topic)
        return sim_list

    def query_sim(self, query_processed, threshold=0.9, top_n=3):
        """
        Performs similarity-based matching of the processed query with topics and returns the most relevant topics.

        Args:
            query_processed (str): The processed query string.
            threshold (float): The similarity threshold for considering a topic as relevant. Defaults to 0.9.
            top_n (int): The number of top relevant topics to return if the similarity is below the threshold. Defaults to 3.

        Returns:
            list: The most relevant topics based on similarity matching with the query.
        """
        sim_list = self.sim_list_maker(query_processed, self.spacy_sim)
        max_sim = max(sim_list)

        if max_sim > threshold:
            max_index = sim_list.index(max_sim)
            sim_topic = self.topics[max_index]

            return [sim_topic]

        elif max_sim < 0.1:
            return ["Irrelevant question"]

        else:
            # Get the indices of the top n most similar texts
            top_indices = sorted(range(len(sim_list)), key=lambda i: sim_list[i], reverse=True)[:top_n]
            # Create a list of the most similar topics
            sim_topics = [self.topics[i] for i in top_indices]

            return sim_topics

    def answer(self, sim_topics):
        """
        Retrieves and displays the answers corresponding to the most relevant topics.

        Args:
            sim_topics (list): The most relevant topics.

        Returns:
            None
        """
        if "Irrelevant question" in sim_topics:
            st.write("Irrelevant question")
        else:
            answers = []
            for i in sim_topics:
                # Define a Cypher query to get all nodes with the given label and property value
                answer_query = f"MATCH (n)-[r:Topic]->(m) WHERE r.data = '{i}' RETURN m.text as options"
                # Run the query and extract information
                res = self.graph.run(answer_query)
                ans_df = pd.DataFrame(res.data())
                answer = list(set(ans_df.options))
                answers.append(answer[0])

            if len(answers) < 2:
                self.print_letter_by_letter(answers[0])
            else:
                radios = sim_topics.copy()
                radios.insert(0, None)
                option = st.radio("Please select the relevant information you wish to know:", options=radios)
                if option in sim_topics:
                    index = sim_topics.index(option)
                    self.print_letter_by_letter(answers[index])

    def process_query(self, query):
        """
        Preprocesses the given query by performing tokenization, lowercasing, stopword removal,
        lemmatization, and punctuation removal.

        Args:
            query (str): The input query to preprocess.

        Returns:
            str: The processed version of the query.
        """
        # Tokenization
        doc_query = self.nlp(query)
        tokens = [token.text for token in doc_query]

        # Lowercasing
        lowercase_tokens = [token.lower() for token in tokens]

        # Stopword Removal
        filtered_tokens = [
            token for token in lowercase_tokens if not self.nlp.vocab[token].is_stop
        ]

        # Lemmatization
        lemmatized_tokens = [token.lemma_ for token in self.nlp(" ".join(filtered_tokens))]

        # Punctuation Removal
        punct_removed_tokens = [
            token for token in lemmatized_tokens if not self.nlp.vocab[token].is_punct
        ]

        # Join the tokens to form the processed query
        query_processed = " ".join(punct_removed_tokens)

        # Return the processed query
        return query_processed

    def print_letter_by_letter(self, text, delay=0.01):
        """
        Prints the given text letter by letter with a specified delay between each letter.

        Args:
            text (str): The text to print.
            delay (float): The delay between each letter in seconds. Defaults to 0.01.

        Returns:
            None
        """
        t = st.empty()
        for i in range(len(text) + 1):
            t.markdown("%s" % text[0:i])
            time.sleep(delay)

    def run(self, query):
        profiler = Profiler()
        profiler.start()

        query_processed = self.process_query(query)
        sim_topics = self.query_sim(query_processed)

        profiler.stop()
        print(profiler.output_text(unicode=True, color=True))

        self.answer(sim_topics)


if __name__ == "__main__":
    ie = InformationExtractor()
    st.markdown("<h1 style='text-align: center; color: grey;'>Information Extractor</h1>", unsafe_allow_html=True)
    query = st.text_input("Ask your question:")
    if query:
        ie.run(query)

