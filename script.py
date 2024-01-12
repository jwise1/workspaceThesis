import streamlit as st
import pandas as pd
import numpy as np
import os 
import dill
import pickle

# with open("model.dill", "rb") as f:
#     model=dill.load(f)

with open("weight.pkl", "rb") as f:
    weight = pickle.load(f)
with open("bias.pkl", "rb") as f:
    bias=pickle.load(f)

def generatePass(lyric):
    return genModel.predict(lyric)

def predictSec(uMonth):
    return str((float(uMonth)*weight)+bias)

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df = df.fillna("None")
    return df

def main():
    # This sets the page title and a cute favicon
    st.set_page_config(page_title='Using Song Lyrics to Generate Memorable Passwords and Analyzing Their Security', page_icon="")

    # st.title("Predicting Monthly Receipt Totals with Linear Regression🧾")

    # # Set a few custom parameters to make our plot blend in with the white background
    # custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    # sns.set_theme(style="ticks", rc=custom_params)
    # sns.color_palette("Set2")

    # Plot the data using Seaborn's countplot
    # fig, ax = plt.subplots(figsize=(30, 8))
    # ax = sns.scatterplot(data,x=data["# Date"],y=data["Receipt_Count"])

    # st.pyplot(fig)

    # input as model generation, ability to edit from user
    lyricInput = st.text_input("Copy/paste or enter song lyrics here to generate a base passphrase.","Type Here")
    basePWord=generatePass(lyricInput)
    fullPWord= st.text_input("Your passphrase from the lyrics is below. Add, remove, or edit the passphrase to make it your own: ", basePWord)

    safe_html ="""  
        <div style="background-color:#80ff80; padding:10px >
        <h2 style="color:white;text-align:center;"> The Abalone is young</h2>
        </div>
        """
    if st.button("Generate security measurement of passphrase."):
        output = predictSec(fullPWord)
        st.success('The security score of your passphrase is {}'.format(str(output)))

if __name__=='__main__':
    main()