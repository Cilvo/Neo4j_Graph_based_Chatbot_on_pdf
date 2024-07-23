from pyinstrument import Profiler
import streamlit as st
import re
from py2neo import Graph, Node , Relationship
import spacy
import fitz
from parrot import Parrot
import warnings
warnings.filterwarnings("ignore")
from transformers import T5ForConditionalGeneration, T5Tokenizer
model = T5ForConditionalGeneration.from_pretrained("Michau/t5-base-en-generate-headline")
tokenizer = T5Tokenizer.from_pretrained("Michau/t5-base-en-generate-headline")

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
    if len(para_list) == 0:
        para_list.append(query)
    # Return the list of paraphrased versions of the query
    return para_list



def michau_transformer_gen_headlines(article):
    """
    Generates headlines for the given article using the Michau Transformer model.

    Args:
        article (str): The article text.

    Returns:
        str: The generated headline.
    """
    encoding = tokenizer.encode_plus(article, return_tensors="pt")
    input_ids = encoding["input_ids"]
    attention_masks = encoding["attention_mask"]

    # Generate headline using the Michau Transformer model
    beam_outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_masks,
        num_beams=3,
        early_stopping=True
    )

    # Decode and clean up the generated headline
    result = tokenizer.decode(beam_outputs[0])
    cleaned_result = result.replace("<pad>", "").replace("</s>","")

    return cleaned_result


def remove_special_characters(text):
    """
    Removes special characters from the given text using regular expressions.

    Args:
        text (str): The input text.

    Returns:
        str: The text with special characters removed.
    """
    # Define the pattern to match special characters (excluding URLs)
    pattern = r'[^a-zA-Z0-9\s\/:.-]'

    # Remove special characters using regular expressions
    cleaned_text = re.sub(pattern, ' ', text)

    return cleaned_text



def extract_table_text(page):
    """
    Extracts the text content from table blocks on a page.

    Args:
        page: The page object representing the document.

    Returns:
        list: A list of text extracted from the table blocks.
    """
    # Get the table blocks on the page
    table_blocks = page.get_text_blocks()

    text_list = []

    # Iterate through each block in the table blocks
    for block in table_blocks:
        # Iterate through each row in the block
        for row in block:
            # Check if the row is a string
            if isinstance(row, str):
                # Replace newline characters with spaces
                row = row.replace('\n', ' ')
                # Check if the row is not empty or consists of only whitespace
                if not row.isspace():
                    text_list.append(row)

    return text_list





def main():

    # Connect to Neo4j
    graph = Graph("bolt://localhost:7689", auth=("neo4j", "password"))
    nlp = spacy.load("en_core_web_sm")

    uploaded_pdf = st.file_uploader(label= "Upload the Document",
                        type= "pdf",
                        accept_multiple_files= False)

    if uploaded_pdf is not None:
        

        # Create a checkbox
        checkbox = st.checkbox("Add Topic")
        text_input = None
        # Check if the checkbox is selected
        if checkbox:
            # Display a text box
            text_input = st.text_input("Enter Topic")
            submit = st.button("Submit")
        if submit:
            profiler = Profiler()
            profiler.start()
            
            pdf_file = fitz.open(stream=uploaded_pdf.read(), filetype="pdf")
            pdf_file_name = uploaded_pdf.name

            # Define the Cypher query
            cypher_query = "MATCH (pdf:PDF) RETURN pdf.name"
            # Execute the query and retrieve the data property values
            try:
                pdf_list = graph.run(cypher_query).to_table()[0]
            except:
                pdf_list = []
            
            if pdf_file_name not in pdf_list:
                # Create a node for the PDF document
                pdf_node = Node("PDF", name=pdf_file_name)
                graph.create(pdf_node)

                for pg_no in range(len(pdf_file)):
                    # Get the current page object
                    page = pdf_file[pg_no]

                    # Extract the text from the page object
                    page_text = page.get_text()

                    if text_input:
                        heading = text_input
                        heading_para = paraphrase(heading)
                    else:
                        heading = michau_transformer_gen_headlines(page_text)
                        heading_para = paraphrase(heading)

                    page_text = page_text.split("\n", 1)
                    page_text = page_text[1] if len(page_text) > 1 else ""

                    page_text = page_text.replace("\n", " ")
                    page_text = remove_special_characters(page_text)

                    doc = nlp(page_text) 

                    page_node = Node("PAGE", id = pg_no, text = page_text)
                    pg_rel = Relationship(pdf_node, 'Topic', page_node, data = heading, heading_list = heading_para)


                    graph.create(page_node)
                    graph.create(pg_rel)

                    if page.get_text_blocks():
                        table_list = extract_table_text(page)
                        table_node = Node("TABLE", id = pg_no, table = table_list)
                        table_relation = Relationship(pdf_node, 'CONTAINS', table_node, data = michau_transformer_gen_headlines(" ".join(table_list)))
                        graph.create(table_node)
                        graph.create(table_relation)
                        

                    sent_after_node = None
                    for sent in doc.sents:
                        sentence_node = Node("Sentence",topic = heading, text= sent.text)
                        sent_rel = Relationship(page_node, "CONTAINS", sentence_node)
                    
                        graph.create(sentence_node)
                        graph.create(sent_rel) 
                        
                        if sent_after_node:
                            sent_after = Relationship(sent_after_node, 'AFTER', sentence_node)
                            graph.create(sent_after)
                        sent_after_node = sentence_node 
                st.write('Successfully Uploaded to Database') 
            else:
                st.write('File Already Exists')

            profiler.stop()

            print(profiler.output_text(unicode=True, color=True))


if __name__ == '__main__':


    main()
