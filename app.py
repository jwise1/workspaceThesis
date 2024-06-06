import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import numpy as np
import keras
import os 
import dill
import pickle
import re
from keras.preprocessing.sequence import pad_sequences

with open('tokenizer.pickle', 'rb') as f:
    sec_tokenizer = pickle.load(f)

secModel=keras.models.load_model("RNNsec_model.keras")

@st.cache_resource
def loadTokenizer():
    return AutoTokenizer.from_pretrained("miscjose/mt5-small-finetuned-genius-music")

@st.cache_resource
def loadModel():
    return AutoModelForSeq2SeqLM.from_pretrained("miscjose/mt5-small-finetuned-genius-music")
    #genModel = AutoModelForSeq2SeqLM.from_pretrained("spiece.model")

#with open("geniusTransformerModel", "rb") as f:
#    genModel=pickle.load(f)

def preprocess(tokenizer,lyric):
    sample = re.sub("[\(\[].*?[\)\]]", "", lyric)
    input_ids = tokenizer(sample, truncation=True,max_length=1024, return_tensors="pt").input_ids
    return input_ids

def generatePass(lyric_,temperature=1):
    tokenizer=loadTokenizer()
    genModel=loadModel()
    lyric=preprocess(tokenizer,lyric_)
    # bracket and parenthesis removal
    # turn to encodings representation
    generation=genModel.generate(lyric,min_length=6,max_length=10,do_sample=True,temperature=float(temperature))
    out=(tokenizer.batch_decode(generation,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0])
    return out

def preprocess2(passphrase):
    passphrase = sec_tokenizer.texts_to_sequences([list(passphrase)])
    passphrase1=pad_sequences(passphrase,maxlen=30)
    return passphrase1

def predictSec(passphrase):
    # reduce password to chars and encode values
    # see j notebook passwordSec
    passphrase1=preprocess2(passphrase)
    return secModel.predict(passphrase1)

# @st.cache_data
# def load_data(file):
#     df = pd.read_csv(file)
#     df = df.fillna("None")
#     return df

def click_button():
    st.session_state.clicked = True

def main():
    # This sets the page title and a cute favicon
    st.set_page_config(page_title='Using Song Lyrics To Generate Memorable Passwords And Analysis Of Their Security', page_icon="")

    st.title("Using Song Lyrics To Generate Memorable Passwords And Analysis Of Their Security🧾")

    # # Set a few custom parameters to make our plot blend in with the white background
    # custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    # sns.set_theme(style="ticks", rc=custom_params)
    # sns.color_palette("Set2")

    # Plot the data using Seaborn's countplot
    # fig, ax = plt.subplots(figsize=(30, 8))
    # ax = sns.scatterplot(data,x=data["# Date"],y=data["Receipt_Count"])

    # st.pyplot(fig)

    # input as model generation, ability to edit from user
    if 'bPW' not in st.session_state:
        st.session_state.clicked = False
        st.session_state['bPW']="Type Here"

    lyricInput = st.text_input("Copy/paste or type song lyrics here to generate a passphrase that you'll then edit.","Type Here")
    
    if st.button("Generate password from lyrics", on_click=click_button):
        basePWord=generatePass(lyricInput)
        
        if "remix" not in lyricInput or "freestyle" not in lyricInput or "song" not in lyricInput:
            basePWord=str(basePWord.replace("remix",""))
            basePWord=str(basePWord.replace("freestyle",""))
            basePWord=str(basePWord.replace("song",""))
        basePWord=str(basePWord.title())
        basePWord=str(basePWord.replace(" ",""))
        st.session_state['bPW']=basePWord
        st.rerun()
    if st.session_state.bPW:
        fullPWord= st.text_input("Your passphrase from the lyrics is below. Add, remove, or edit the passphrase to make it your own: ",st.session_state['bPW'])
        if st.button("Generate security score of passphrase"):
            output=predictSec(fullPWord)
            output=output[0]
            if round(output[0])==1:
                st.success('The security score of your passphrase (0) is low. Consider adding special characters and further editing your passphrase.')
            elif round(output[1])==1:
                st.success('The security score of your passphrase (1) is medium. You can make your passphrase more secure by adding special characters and further editing.')
            elif round(output[2])==1:
                st.success('The security score of your passphrase (2) is high. Your secure passphrase is ready for use!')

if __name__=='__main__':
    main()