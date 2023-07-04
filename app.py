import os
from apikey import openai_api_key

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper


os.environ['OPENAI_API_KEY'] = openai_api_key

st.title("🦜️🔗 Youtbe GPT Content Creator")
prompt = st.text_input('Plug in your prompt here')

title_template = PromptTemplate(
    input_variables= ['topic'], 
    template= 'Write me a youtube video title about {topic}'

)

script_template = PromptTemplate(
    input_variables= ['title', 'wikipedia_research'], 
    template= 'Write me a youtube video script about title TITLE: {title} while leveraging on wikipedia research: {wikipedia_research}' 

)

title_memory = ConversationBufferMemory(input_key= 'topic', memory_key= 'chat_history')
script_memory = ConversationBufferMemory(input_key= 'title', memory_key= 'chat_history')

llms = OpenAI(temperature=0)
title_chain = LLMChain(llm=llms, prompt=title_template, output_key= 'title',  memory=title_memory)
script_chain = LLMChain(llm=llms, prompt=script_template, output_key= 'script', memory=script_memory)


wiki = WikipediaAPIWrapper()
if prompt:
    title = title_chain.run(prompt)
    wiki_research  = wiki.run(prompt)
    script = script_chain.run(title = title, wikipedia_research = wiki_research)
    st.write(title)
    st.write(script)

    
    with st.expander('Title Histry'):
        st.info(title_memory.buffer)
        
    with st.expander('Script Histry'):
        st.info(script_memory.buffer)

        
    with st.expander('Wikipedia Research Histry'):
        st.info(wiki_research)


    
    
    