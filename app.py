import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers

#Function to fetch response from the LLAMA 2 Model


def getResponseFromLLama(input_text, number_of_words, blog_type):
    
    # Call the LLAMA Model by firstly downloading from HUGGING FACE and then using it
    llm= CTransformers(model='models/llama-2-7b-chat.ggmlv3.q8_0.bin',
                      model_type='llama',
                      config={'max_new_tokens':256,
                              'temperature':0.01})

    # Make the prompt template

    template = f"""
    Write a blog for {blog_type} job profile for a topic {input_text}
    within {number_of_words} words.
                """

    prompt = PromptTemplate(input_variables=["style", "text", "no-of-words"],
                            template=template)
    
    # Generate the response from the LLAMA 2 Model

    response = llm(prompt.format(style=blog_type, text = input_text, n_words = number_of_words )) 
    print(response)
    return response



st.set_page_config(page_title="Generate Blogs",
                    page_icon='🤖',
                    layout='centered',
                    initial_sidebar_state='collapsed')

st.header("Make Blogs 🤖")
input_text = st.text_input("Enter topic for discussion in BLOG")

# Creating column for additional 2 fields

col1,col2 = st.columns([5,5])

with col1:
    number_of_words = st.text_input('No of words')
with col2:
    blog_type = st.selectbox('Creating the blog for',('Data Science guys','Research people', 'General Public' ), index = 0)

submit = st.button("Generate")

# Actual response

if submit:
    st.write(getResponseFromLLama(input_text, number_of_words, blog_type))





