import streamlit as st
from functions import process, query_sim4, answer4

from pyinstrument import Profiler
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
