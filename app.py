import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
import pandas as pd
import numpy as np
import keras
import os 
import pickle
import re
import tensorflow
import random
import string
from nltk.util import ngrams
import time
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

@st.cache_resource
def loadSecTokenizer():
    with open('tokenizer.pickle', 'rb') as f:
        sec_tokenizer = pickle.load(f)
    return sec_tokenizer

@st.cache_resource
def loadSecModel():
    secModel=tensorflow.keras.models.load_model("./secModelv3.keras")
    return secModel

@st.cache_resource
def loadTokenizer():
    return AutoTokenizer.from_pretrained("miscjose/mt5-small-finetuned-genius-music")

@st.cache_resource
def loadModel():
    return AutoModelForSeq2SeqLM.from_pretrained("./myGenModel6/")
    #genModel = AutoModelForSeq2SeqLM.from_pretrained("spiece.model")

@st.cache_resource
def promptLLM(system_prompt):
    llm = Ollama(model="llama3")
    template = """
        <|begin_of_text|>
        <|start_header_id|>system<|end_header_id|>
        {system_prompt}
        <|eot_id|>
        """
    prompt = PromptTemplate(
        input_variables=["system_prompt"],
        template=template
    )
    response = llm.invoke(prompt.format(system_prompt=system_prompt))
    return response


def loadData():
    trigrams=pd.read_csv("./data/passSecData/3grams_english.csv",usecols=["ngram"])
    qgrams=pd.read_csv("./data/passSecData/4grams_english.csv",usecols=["ngram"])
    fivegrams=pd.read_csv("./data/passSecData/5grams_english.csv",usecols=["ngram"])
    grams=pd.concat([trigrams,qgrams],axis=0)
    allgrams=pd.concat([grams,fivegrams],axis=0)
    return allgrams

def preprocess(tokenizer,lyric):
    sample = re.sub("[\(\[].*?[\)\]]", "", lyric)
    input_ids = tokenizer(sample, truncation=True,max_length=512, return_tensors="pt").input_ids
    return input_ids

def generatePass(lyric_,temperature=1):
    tokenizer=loadTokenizer()
    genModel=loadModel()
    lyric=preprocess(tokenizer,lyric_)
    # bracket and parenthesis removal
    # turn to encodings representation
    generation=genModel.generate(lyric,min_length=5,max_length=8,do_sample=True,temperature=float(temperature))
    out=(tokenizer.batch_decode(generation,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0])
    return out

def preprocess2(passphrase):
    sec_tokenizer=loadSecTokenizer()
    passphrase = sec_tokenizer.texts_to_sequences([list(passphrase)])
    passphrase1=pad_sequences(passphrase,maxlen=30)
    return passphrase1

def predictSec(passphrase):
    # reduce password to chars and encode values
    secModel=loadSecModel()
    passphrase1=preprocess2(passphrase)
    return secModel.predict(passphrase1)

def processText(lyrics):
    if "\n" in lyrics:
        lyrics = re.sub("[\(\[].*?[\)\]]", "", lyrics)
        lyrics=lyrics.translate(str.maketrans('', '', string.punctuation)).lower()
        l=lyrics.split('\n')
        arr = [str(x) for x in l if x]
        n=random.randint(0,len(arr)-1)
        lyrics=" ".join(arr[n:n+2])
    else:
        count=len(re.findall(r'\w+', lyrics))
        try:
            f=(count/12)>0
            n=random.randint(0,count-16)
            lyrics=" ".join(lyrics.split(" ")[n:n+16])
        except:
            if len(re.findall(r'\w+',lyrics))>6:
                lyrics=lyrics
            else:
                st.success("You must enter a longer lyric.")
    return lyrics

def checkNGrams(text):
    data1=loadData()
    for x in range(0,len(data1.values)):
        if str(data1.values[x][0]).lower().strip().replace(" ' ","").replace(" '","").replace("' ","").replace("'", "").replace('"','').replace(" ,","").replace(",","") in text.lower().strip():
            return False
        else:
            continue
    return True

def click_button():
    st.session_state.clicked = True

def main():
    # This sets the page title and a cute favicon
    st.set_page_config(page_title='Using Song Lyrics To Generate Memorable Passwords And Analysis Of Their Security', page_icon="")

    st.title("Using Song Lyrics To Generate Memorable Passwords And Analysis Of Their Security🧾")

    # input as model generation, ability to edit from user
    if 'bPW' not in st.session_state or 'bPW1' not in st.session_state:
        st.session_state.clicked = False
        st.session_state['bPW']="Type Here"
        st.session_state['bPW1']=" "

    lyricInput = st.text_area("Copy/paste or type song lyrics line by line here to generate a passphrase that you'll then edit. Please note, the password generated is meant to be personalized and added onto by you for memorability and security.","Type Here")
    
    # temperature functions--if temp=2 remove temp=1 prediction, etc.
    # longer generations
    temp=st.slider(label="Variations in Text Generated:",min_value=1,max_value=3,value=1)
    check1=st.checkbox(label="Include only words found in lyrics")
    
    if st.button("Generate Password", on_click=click_button):

        if check1==True:
                if temp==1:
                    # random two lines
                    line=processText(lyricInput)
                    basePWord=generatePass(line)
                    start=time.time()
                    # check if contained in lyric/check n-grams
                    while(basePWord.strip() not in line.strip() or checkNGrams(basePWord.strip())!=True or len(set(basePWord.strip().split(" ")))<=3):
                        basePWord=generatePass(line)
                        if time.time()-start > 1.5 or len(set(basePWord.strip().split(" ")))<=3:
                            #print(time.time()-start)
                            line=processText(lyricInput)
                            basePWord=generatePass(line)
                # TEMP 2 and 3 NEED WORK
                if temp>=2:
                    # random two lines
                    line=processText(lyricInput)
                    basePWord=generatePass(line)
                    start=time.time()
                    # check if contained in lyric/check n-grams
                    while(basePWord.strip() not in line.strip()):
                        basePWord=generatePass(line)
                        if time.time()-start > 2.5 or len(set(basePWord.strip().split(" ")))<=3:
                            #print(time.time()-start)
                            line=processText(lyricInput)
                            basePWord=generatePass(line)
                    # random two lines
                    lyricInput2=line.replace(basePWord,"")
                    lyricInput2=lyricInput2.replace("  "," ")
                    print(line)
                    print(lyricInput2)
                    #line=processText(lyricInput2)
                    basePWord=generatePass(lyricInput2)
                    start=time.time()

                    # check if contained in lyric/check n-grams
                    while(checkNGrams(basePWord.strip())!=True):
                        basePWord=generatePass(lyricInput2)
                        #if time.time()-start > 2.5 or len(set(basePWord.strip().split(" ")))<=3: 
                            #print(time.time()-start)
                            #line=processText(lyricInput2)
                            #basePWord=generatePass(lyricInput2)
                    # if temp==3:
                    #     lyricInput3=lyricInput2.replace(basePWord,"")
                    #     # random two lines
                    #     #line=processText(lyricInput3)
                    #     basePWord=generatePass(lyricInput3)
                    #     start=time.time()
                    #     # check if contained in lyric/check n-grams
                    #     while(checkNGrams(basePWord.strip())!=True or len(set(basePWord.strip().split(" ")))<=3):
                    #         basePWord=generatePass(lyricInput3)
                    #         if time.time()-start > 3.5 or len(set(basePWord.strip().split(" ")))<=3:
                    #             #print(time.time()-start)
                    #             #line=processText(lyricInput3)
                    #             basePWord=generatePass(lyricInput3)                       
        else:
            line=processText(lyricInput)
            print(line)
            basePWord=generatePass(line,temperature=temp)
            start=time.time()
            while(checkNGrams(basePWord.strip())!=True or len(set(basePWord.strip().split(" ")))<=3):
                basePWord=generatePass(line,temperature=temp)
                if time.time()-start > 2.5 or len(set(basePWord.strip().split(" ")))<=3:
                    #print(time.time()-start)
                    line=processText(lyricInput)
                    print(line)
                    basePWord=generatePass(line,temperature=temp)
        # capture surrounding characters
        # flag=True
        # while(flag):
        #     line=line.strip()
        #     basePWord=basePWord.strip()
        #     try:
        #         if line.startswith(line[line.find(basePWord)])!=True and line[line.find(basePWord)-1]!=" ":
        #             basePWord1=line[line.find(basePWord)-1]+basePWord
        #             basePWord=basePWord1   
        #         else:
        #             flag=False
        #             break
        #     except:
        #         break
        # flag=True
        # while(flag):
        #     line=line.strip()
        #     basePWord=basePWord.strip()
        #     try:
        #         if line.endswith(line[line.find(basePWord)+len(basePWord)])!=True and line[line.find(basePWord)+len(basePWord)]!=" ":
        #             basePWord=basePWord+line[line.find(basePWord)+len(basePWord)]
        #         else:
        #             flag=False
        #             break
        #     except:
        #         break
        basePWord=str(basePWord.title())
        basePWord=str(basePWord.replace(" ","_"))
        st.session_state['bPW1']=basePWord
        st.rerun()
    if st.session_state.bPW:
        fullPWord= st.text_input("Your base passphrase from the lyrics is below. Using the song lyrics generated alone is not recommended. Add complexity by substituting letters with characters, adding numbers, inserting special characters, and using proper nouns either randomly or relatable to you to make it more memorable and secure: ",st.session_state['bPW1'])
        if st.button("Generate Security Score of Password"):

            output=predictSec(fullPWord.replace("_",""))
            output=output[0]
            regex = re.compile('[@_!#$%^&*()<>?/\|}{~:]')
            if round(output[0])==1:
                st.success('The security score of your passphrase (0.01 out of 2) is low. Consider adding special characters and further editing your passphrase. You can add numbers and words meaningful to you to improve security and memorability.')
            elif round(output[1])==1 and regex.search(fullPWord.replace("_",""))!=None:
                st.success('The security score of your passphrase (1 out of 2) is medium. You can make your passphrase more secure by adding special characters and numbers, as well as adding personally relatable words.')
            elif round(output[1])==1 and regex.search(fullPWord.replace("_",""))==None:
                st.success('The security score of your passphrase (1 out of 2) is medium. Be sure to make your passphrase more secure by adding special characters and not making it too short.')
            elif round(output[2])==1 and regex.search(fullPWord.replace("_",""))!=None:
                st.success('The security score of your passphrase (2 out of 2) is high. Your secure passphrase is ready for use!')
            elif round(output[2])==1 and regex.search(fullPWord.replace("_",""))==None:
                st.success('The security score of your passphrase (2 out of 2) is high, but you should add more special characters and numbers to it to make it more unique.')
            # needs implementation
        if st.button("Generate Story from Password"):
            st.success(promptLLM("Generate a unique story from the following phrase using proper nouns from the phrase WITHOUT repeating the phrase in the story and WITHOUT including the same sequence of words in the story: "+fullPWord))
            


if __name__=='__main__':
    main()