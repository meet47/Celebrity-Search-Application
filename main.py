#Integrate code with OpenAI API
import os
from constants import openai_key
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate 
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory 

import streamlit as st

os.environ["OPENAI_API_KEY"] = openai_key

#Steamlit Framework
st.title('Celebrity Search Application')
input_text = st.text_input("Search about your favourite celebrity")

#OpenAI LLMs
llm = OpenAI(temperature=0.8)

#Prompt Template
first_input_prompt = PromptTemplate(
    input_variables = ['name'],
    template = "Tell me about celebrity {name}"
)

person_memory = ConversationBufferMemory(input_key='name', memory_key='chat_history')
chain = LLMChain(llm=llm, prompt=first_input_prompt, verbose=True, output_key='person', memory=person_memory)


second_input_prompt = PromptTemplate(
    input_variables = ['person'],
    template = "when was {person} born"
)

dob_memory = ConversationBufferMemory(input_key='person', memory_key='chat_history')
chain2 = LLMChain(llm=llm, prompt=second_input_prompt, verbose=True, output_key='dob', memory=dob_memory)

third_input_prompt = PromptTemplate(
    input_variables = ['dob'],
    template = "Mention 5 major events happended around {dob} in the world"
)

description_memory = ConversationBufferMemory(input_key='dob', memory_key='chat_history')
chain3 = LLMChain(llm=llm, prompt=third_input_prompt, verbose=True, output_key='description', memory=description_memory)

parent_chain = SequentialChain(chains=[chain, chain2, chain3], 
                               input_variables=['name'], 
                               output_variables=['person', 'dob', 'description'],
                               verbose=True)

if input_text:
    st.write(parent_chain({'name':input_text}))
    with st.expander('Person Name'):
        st.info(person_memory.buffer)

    with st.expander('Major Events'):
        st.info(description_memory.buffer)