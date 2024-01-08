#!/usr/bin/env python
# coding: utf-8

# In[3]:


pip install unstructured


# In[4]:


pip install "unstructured[all-docs]"


# In[5]:


pip install "unstructured[local-inference]"


# In[6]:


pip install unstructured==0.5.6


# In[39]:


from langchain.document_loaders import DirectoryLoader

directory ='C:\\Users\\msgupta9\\PycharmProjects\\pythonProject6\\test2'

def load_docs(directory):
  loader = DirectoryLoader(directory)
  documents = loader.load()
  return documents

documents = load_docs(directory)
len(documents)
print(documents)


# In[3]:


from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_docs(documents,chunk_size=500,chunk_overlap=0):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

docs = split_docs(documents)
print(len(docs))


# In[29]:


from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.embeddings import SentenceTransformerEmbeddings
import openai
from langchain.chains.question_answering import load_qa_chain
openai.api_key = "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"
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma.from_documents(
            documents=docs, embedding=embeddings
        )
llm = ChatOpenAI(model_name="gpt-4-32k", temperature=0,
                         openai_api_key="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")
chain = load_qa_chain(llm, chain_type="stuff", verbose=True)


# In[23]:


query = "Who is the Global Chief Executive Officer of Mastek?"
matching_docs = vectordb.similarity_search(query)
answer = chain.run(input_documents=matching_docs, question=query)
print(answer)


# In[35]:


query = "What was the revenue for Q1FY23?"
matching_docs = vectordb.similarity_search(query)
answer = chain.run(input_documents=matching_docs, question=query)
print(answer)


# In[1]:


import openai
import os
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import UnstructuredExcelLoader



openai.api_key = ""
get_me_data_path = 'C:\\Users\\msgupta9\\PycharmProjects\\pythonProject6\\test2'


# In[2]:


def load_docs(directory):
    try:
        loader = DirectoryLoader(directory)
        print("Loading Directory")
        documents = loader.load()
        print("Directory Loaded")
        return documents
    except Exception as e:
        print("An error occurred:", str(e))


# In[3]:


def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs


# In[4]:


def find_answer(user_question):
        documents = load_docs(get_me_data_path)
        print(f"In this directory we have {len(documents)} files")
        docs = split_docs(documents)
        # print(docs)
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        persist_directory = "chroma_db_data"
        print("Adding to chromadb")
        vectordb = Chroma.from_documents(
            documents=docs, embedding=embeddings, persist_directory=persist_directory
        )
        print("Added data to chromadb")
        vectordb.persist()

        os.environ["OPENAI_API_KEY"] = ""
        llm = ChatOpenAI(model_name="gpt-4-32k", temperature=0,
                         openai_api_key=""
        chain = load_qa_chain(llm, chain_type="stuff", verbose=True)
        system_prompt = """You are an assistant chatbot. Your purpose is to help management consultants at Deloitte. You are an expert at developing business case models. Specifically, you work on Application Modernization business cases, which fall under Tech Cost and Performance Management. Help the user think through tasks and questions.

Your answers must be based on the context provided, which will contain values from a database and/or a table, and documents that contain industry benchmarks. The user will ask you to find specific benchmarks or data points such as “What is the typical labor rate for roles needed for application modernization?” To support the business case.

Your response should provide a clear and concise data point and/or assumptions. Ensure that the data point and/or assumptions are relevant, accurate. Make sure to provide additional context around the data point and/or benchmarks and assumptions with a description.

Be truthful in your answer, if a sufficient answer isn’t found to a user’s chat, respond “I don’t know”. Make sure you refer to the given context to respond to the user."""
        query = user_question
        matching_docs = vectordb.similarity_search(query)
        answer = chain.run(input_documents=matching_docs, question=query, prompt=system_prompt)
        print(answer)


# In[5]:


find_answer('What was the revenue for Q1FY23?')


# In[7]:


find_answer("What was the revenue from operations difference between Q1FY23 and Q4FY22?")


# In[8]:


find_answer("What was the net profit difference between Q1FY23 and Q1FY22?")


# In[9]:


find_answer('What was the operating EBITDA margin for the full year FY23?')


# In[10]:


find_answer('What was the Q4 FY23 Revenue?')


# In[11]:


find_answer('Who is the Auditor of Mastek?')


# In[12]:


find_answer('How many new clients did the company add in Q1FY23?')


# In[13]:


find_answer('How many employees were based offshore in India as of 30th June, 2022? ')


# In[14]:


find_answer(' How many large frameworks has Mastek been shortlisted for in the UK Public Sector?')


# In[ ]:




