from flask import Flask, render_template, request, jsonify
import os
import torch
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate

app = Flask(__name__)

# Load the embedding model
model_name = 'hkunlp/instructor-base'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedding_model = HuggingFaceInstructEmbeddings(
    model_name=model_name,
    model_kwargs={"device": device}
)

# Load the vector store
vector_path = 'vector-store/personal_info'
vectordb = FAISS.load_local(
    folder_path=vector_path,
    embeddings=embedding_model,
    index_name='personal',
    allow_dangerous_deserialization=True
)

# Load the tokenizer and model
model_id = 't5-large'
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_id,
    device_map='cpu'
)

pipe = pipeline(
    task="text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    model_kwargs={
        "temperature": 0,
        "repetition_penalty": 1.5
    }
)

llm = HuggingFacePipeline(pipeline=pipe)

prompt_template = """
    I'm MookBot, your friendly assistant, here to answer any personal questions you have about my background and experiences. 
    Whether you're curious about my age, education, career, or personal beliefs, feel free to ask, 
    and I'll provide answers to help you learn more about me. 
    Just let me know what you're wondering about, and I'll do my best to share.
    {context}
    Question: {question}
    Answer:
    """.strip()

PROMPT = PromptTemplate.from_template(template=prompt_template)

# Create the question generator chain
question_generator = LLMChain(
    llm=llm,
    prompt=CONDENSE_QUESTION_PROMPT,
    verbose=True
)

# Create the document chain
doc_chain = load_qa_chain(
    llm=llm,
    chain_type='stuff',
    prompt=PROMPT,
    verbose=True
)

# Create the memory
memory = ConversationBufferWindowMemory(
    k=3,
    memory_key="chat_history",
    return_messages=True,
    output_key='answer'
)

# Create the conversational retrieval chain
chain = ConversationalRetrievalChain(
    retriever=vectordb.as_retriever(),
    question_generator=question_generator,
    combine_docs_chain=doc_chain,
    return_source_documents=True,
    memory=memory,
    verbose=True,
    get_chat_history=lambda h: h
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['message']
    response = chain({"question": user_input})
    return jsonify({
        "answer": response['answer'],
        "source_documents": [doc.metadata['source'] for doc in response['source_documents']]
    })

if __name__ == '__main__':
    app.run(debug=True)