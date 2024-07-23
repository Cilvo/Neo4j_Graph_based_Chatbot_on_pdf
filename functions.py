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



# # Load the pre-trained BART model and tokenizer
# model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
# tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

# Load the spacy model
nlp = spacy.load("en_core_web_md")

# Create a Graph object and connect to the database
graph = Graph("bolt://localhost:7689", auth=("neo4j", "password"))

# Define a Cypher query to get all nodes with the given label and property value
query = f"MATCH (n)-[r:Topic]->(m) RETURN r.data AS headline, r.heading_list as list"

# Run the query and extract information
results = graph.run(query)
df = pd.DataFrame(results.data())
topics = list(set(df.headline))
topics_paras = list(df.list)

#Init models (make sure you init ONLY once if you integrate this to your code)
parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5", use_gpu=True)


def paraphrase(query):
    """
    Paraphrases the given query using the Parrot paraphraser model.

    Args:
        query (str): The input query to paraphrase.

    Returns:
        list: A list of paraphrased versions of the query.
    """
    # Use the Parrot paraphraser to augment the input phrase (query)
    para_phrases = parrot.augment(input_phrase=query)

    # Initialize an empty list to store the paraphrased versions
    para_list = []

    # Check if any paraphrases are generated
    if para_phrases is not None:
        # Iterate through each paraphrase and append it to the para_list
        for para_phrase in para_phrases:
            para_list.append(para_phrase[0])

    # Return the list of paraphrased versions of the query
    return para_list





def process(query):
    """
    Preprocesses the given query by performing tokenization, lowercasing, stopword removal,
    lemmatization, and punctuation removal.

    Args:
        query (str): The input query to preprocess.

    Returns:
        str: The processed version of the query.
    """
    # Tokenization
    doc_query = nlp(query)
    tokens = [token.text for token in doc_query]

    # Lowercasing
    lowercase_tokens = [token.lower() for token in tokens]

    # Stopword Removal
    filtered_tokens = [
        token for token in lowercase_tokens if not nlp.vocab[token].is_stop
    ]

    # Lemmatization
    lemmatized_tokens = [token.lemma_ for token in nlp(" ".join(filtered_tokens))]

    # Punctuation Removal
    punct_removed_tokens = [
        token for token in lemmatized_tokens if not nlp.vocab[token].is_punct
    ]

    # Join the tokens to form the processed query
    query_processed = " ".join(punct_removed_tokens)

    # Return the processed query
    return query_processed



topics_preprocessed = [process(topic) for topic in topics]


import time
import streamlit as st

def print_letter_by_letter(text, delay=0.01):
    """
    Prints the given text letter by letter with a specified delay between each letter.

    Args:
        text (str): The text to be printed.
        delay (float, optional): The delay in seconds between each letter. Defaults to 0.01.
    """
    t = st.empty()
    for i in range(len(text) + 1):
        t.markdown("%s" % text[0:i])
        time.sleep(delay)



def spacy_sim(query, text):
    """
    Calculates the similarity score between the given query and text using spaCy's similarity measure.

    Args:
        query (str): The query string.
        text (str): The text string to compare with the query.

    Returns:
        float: The similarity score between the query and text.
    """
    doc1 = nlp(query)
    doc2 = nlp(text)
    sim = doc1.similarity(doc2)
    return sim


def sent_trans(query, text):
    """
    Calculates the similarity score between the given query and text using Sentence-Transformers.

    Args:
        query (str): The query string.
        text (str): The text string to compare with the query.

    Returns:
        float: The similarity score between the query and text.
    """
    sentrans = SentenceTransformer("distilbert-base-nli-mean-tokens")
    
    sentences = [query, text]

    # Encode the sentences using Sentence-Transformers
    sentence_embeddings = sentrans.encode(sentences)

    # Calculate the cosine similarity between the sentence embeddings
    similarity_score = 1 - distance.cosine(sentence_embeddings[0], sentence_embeddings[1])

    return similarity_score



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def tf_idf(query, text):
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



def sim_list_maker(query_processed, sim_func):
    """
    Generates a list of similarity scores between the processed query and a list of topics using a specified similarity function.

    Args:
        query_processed (str): The processed query string.
        sim_func (function): The similarity function to calculate the similarity scores.

    Returns:
        list: A list of similarity scores between the query and topics.
    """
    sim_list = []
    for topic in topics:
        sim = sim_func(query_processed, topic)
        sim_list.append(sim)
    return sim_list




def sim_list_maker_paraphrase(query_processed, sim_func):
    '''
    Calculates the similarity scores between the paraphrased queries and the topics using a specified similarity function.

    Args:
        query_processed (str): The processed query string.
        sim_func (function): The similarity function to calculate the similarity scores.

    Returns:
        list: A list of similarity scores between the paraphrased queries and topics.
    '''
    queries = paraphrase(query_processed)
    sim_list = []
    for topic_para in topics_paras:
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




# # =========== Top similarity answer  ===========

# def query_sim(query_processed):
#     sim_list = sim_list_maker(query_processed, spacy_sim)
#     max_sim = max(sim_list)
#     max_index = sim_list.index(max_sim)
#     sim_topic = topics[max_index]
#     return sim_topic


# def answer(sim_topic):
#     # Define a Cypher query to get all nodes with the given label and property value
#     answer_query = (
#         f"MATCH (n)-[r]->(m) WHERE r.data = '{sim_topic}' RETURN m.text as heading"
#     )
#     # Run the query and extract information
#     res = graph.run(answer_query)
#     ans_df = pd.DataFrame(res.data())
#     answer = set(ans_df.heading)

#     text = list(answer)[0]

#     st.write(print_letter_by_letter(text))



# # =========== Top 3 similarity scores and summarizer===========

# def query_sim3(query_processed, n=3):
#     sim_list = sim_list_maker(query_processed, spacy_sim)
#     # Get the indices of the top n most similar texts
#     top_indices = sorted(range(len(sim_list)), key=lambda i: sim_list[i], reverse=True)[
#         :n
#     ]
#     # Create a list of the most similar texts
#     top_topics = [topics[i] for i in top_indices]
#     return top_topics


# def answer3(top_topics):
#     text = ""
#     for i in top_topics:
#         # Cypher query to get all nodes with the given label and property value
#         answer_query = (
#             f"MATCH (n)-[r]->(m) WHERE r.data = '{i}' RETURN m.text as options"
#         )
#         # Run the query and extract information
#         res = graph.run(answer_query)
#         ans_df = pd.DataFrame(res.data())
#         answer = list(set(ans_df.options))

#         text += answer[0]
#     inputs = tokenizer.encode(text, return_tensors="pt")

#     # Generate the summary
#     summary_ids = model.generate(inputs, num_beams=4, early_stopping=True)
#     summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

#     st.write(print_letter_by_letter(summary))




# =========== Threshold else top similarities ===========

def query_sim4(query_processed):
    """
    Performs similarity-based matching of the processed query with topics and returns the most relevant topics.

    Args:
        query_processed (str): The processed query string.

    Returns:
        list: The most relevant topics based on similarity matching with the query.
    """
    sim_list = sim_list_maker(query_processed, spacy_sim)
    max_sim = max(sim_list)
    threshold = 0.9

    if max_sim > threshold:
        max_index = sim_list.index(max_sim)
        sim_topic = topics[max_index]

        return [sim_topic]
    
    elif max_sim < 0.1:
        st.write("Irrelevant question")
        return 0

    else:
        # Get the indices of the top n most similar texts
        top_indices = sorted(range(len(sim_list)), key=lambda i: sim_list[i], reverse=True)[:3]
        # Create a list of the most similar topics
        sim_topics = [topics[i] for i in top_indices]
        
        return sim_topics



def answer4(sim_topics):
    """
    Retrieves and displays the answers corresponding to the most relevant topics.

    Args:
        sim_topics (list): The most relevant topics.

    Returns:
        None
    """
    if sim_topics != 0:
        answers = []
        for i in sim_topics:
            # Define a Cypher query to get all nodes with the given label and property value
            answer_query = f"MATCH (n)-[r:Topic]->(m) WHERE r.data = '{i}' RETURN m.text as options"
            # Run the query and extract information
            res = graph.run(answer_query)
            ans_df = pd.DataFrame(res.data())
            answer = list(set(ans_df.options))
            answers.append(answer[0])

        if len(answers) < 2:
            print_letter_by_letter(answers[0])
        else:
            radios = sim_topics.copy()
            radios.insert(0, None)
            option = st.radio("Please select the relevant information you wish to know:", options=radios)
            if option in sim_topics:
                index = sim_topics.index(option)
                print_letter_by_letter(answers[index])




profiler = Profiler()
profiler.start()

st.markdown("<h1 style='text-align: center; color: grey;'>Information Extractor</h1>", unsafe_allow_html=True)

query = st.text_input("Ask the question:")

if query:
    query_processed = query
    sim_topic = query_sim4(query_processed)
    answer4(sim_topic)

profiler.stop()

print(profiler.output_text(unicode=True, color=True))
