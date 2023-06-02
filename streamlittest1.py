#!/usr/bin/env python
# coding: utf-8

# In[2]:

import numpy as np
import streamlit as st
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("tblard/tf-allocine",use_fast= False)
model = TFAutoModelForSequenceClassification.from_pretrained("tblard/tf-allocine")

nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

def analyze_sentiment(sentiment):
    result=nlp(sentiment)[0]
    return result
def main():
    st.title("Analyse de sentiment")
    st.write("Entrez un sentiment pour l'analyser :")
    sentiment = st.text_input("Sentiment")

    if st.button("Analyser"):
        result = analyze_sentiment(sentiment)
        polarity = np.array(result['label'])
        score=np.array(result['score'])
        st.write("Le sentiment est :", np.array2string(polarity), " à ",np.array2string(score)," de certitude")

if __name__ == "__main__":
    main()


# In[ ]: