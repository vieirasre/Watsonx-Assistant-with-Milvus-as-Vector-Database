from datasets import load_dataset
from IPython.display import clear_output
import pandas as pd
import re
from dotenv import load_dotenv
import os
from pymilvus import Collection, utility
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from towhee import pipe, ops
import numpy as np
from towhee.datacollection import DataCollection
from typing import List

from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods
from langchain.llms import WatsonxLLM
#import langchain.chains as lc
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

print_full_prompt=False

## Step 1 Dataset Retrieving

dataset = load_dataset("ruslanmv/ai-medical-chatbot")
clear_output()
train_data = dataset["train"]
#For this demo let us choose the first 1000 dialogues

df = pd.DataFrame(train_data[:1000])
#df = df[["Patient", "Doctor"]].rename(columns={"Patient": "question", "Doctor": "answer"})
df = df[["Description", "Doctor"]].rename(columns={"Description": "question", "Doctor": "answer"})
# Add the 'ID' column as the first column
df.insert(0, 'id', df.index)
# Reset the index and drop the previous index column
df = df.reset_index(drop=True)

# Clean the 'question' and 'answer' columns
df['question'] = df['question'].apply(lambda x: re.sub(r'\s+', ' ', x.strip()))
df['answer'] = df['answer'].apply(lambda x: re.sub(r'\s+', ' ', x.strip()))
df['question'] = df['question'].str.replace('^Q.', '', regex=True)
# Assuming your DataFrame is named df
max_length = 500  # Due to our enbeeding model does not allow long strings
df['question'] = df['question'].str.slice(0, max_length)
#To use the dataset to get answers, let's first define the dictionary:
#- `id_answer`: a dictionary of id and corresponding answer
id_answer = df.set_index('id')['answer'].to_dict()

def create_milvus_collection(collection_name, dim):
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
    fields = [
    FieldSchema(name='id', dtype=DataType.VARCHAR, descrition='ids', max_length=500, is_primary=True, auto_id=False),
    FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, descrition='embedding vectors', dim=dim)
    ]
    schema = CollectionSchema(fields=fields, description='reverse image search')
    collection = Collection(name=collection_name, schema=schema)
    # create IVF_FLAT index for collection.
    index_params = {
        'metric_type':'L2',
        'index_type':"IVF_FLAT",
        'params':{"nlist":2048}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    return collection

COLLECTION_NAME='qa_medical'
load_dotenv()
host_milvus = os.environ.get("REMOTE_SERVER", '127.0.0.1')
connections.connect(host=host_milvus, port='19530')

insert_pipe = (
    pipe.input('id', 'question', 'answer')
        .map('question', 'vec', ops.text_embedding.dpr(model_name='facebook/dpr-ctx_encoder-single-nq-base'))
        .map('vec', 'vec', lambda x: x / np.linalg.norm(x, axis=0))
        .map(('id', 'vec'), 'insert_status', ops.ann_insert.milvus_client(host=host_milvus, port='19530', collection_name='question_answer'))
        .output()
)

#collection = create_milvus_collection(COLLECTION_NAME, 768)
# for index, row in df.iterrows():
#     insert_pipe(str(row['id']), row['question'], row['answer'])

## Step 2 WatsonX connection

load_dotenv()
try:
    API_KEY = os.environ.get("API_KEY")
    project_id =os.environ.get("PROJECT_ID")
except KeyError:
    API_KEY: input("Please enter your WML api key (hit enter): ")
    project_id  = input("Please  project_id (hit enter): ")

credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey": API_KEY  
}    

model_id = ModelTypes.GRANITE_13B_CHAT_V2


parameters = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.MAX_NEW_TOKENS: 500,
    GenParams.STOP_SEQUENCES: ["<|endoftext|>"]
}


watsonx_granite = WatsonxLLM(
    model_id=model_id.value,
    url=credentials.get("url"),
    apikey=credentials.get("apikey"),
    project_id=project_id,
    params=parameters
)


## Step 3 Milvus connection

collection = Collection(COLLECTION_NAME)      
collection.load(replica_number=1)
utility.load_state(COLLECTION_NAME)
utility.loading_progress(COLLECTION_NAME)


max_input_length = 500  # Maximum length allowed by the model



# Create the combined pipe for question encoding and answer retrieval
combined_pipe = (
    pipe.input('question')
        .map('question', 'vec', lambda x: x[:max_input_length])  # Truncate the question if longer than 512 tokens
        .map('vec', 'vec', ops.text_embedding.dpr(model_name='facebook/dpr-ctx_encoder-single-nq-base'))
        .map('vec', 'vec', lambda x: x / np.linalg.norm(x, axis=0))
        .map('vec', 'res', ops.ann_search.milvus_client(host=host_milvus, port='19530', collection_name=COLLECTION_NAME, limit=1))
        .map('res', 'answer', lambda x: [id_answer[int(i[0])] for i in x])
        .output('question', 'answer')
)
   
# Step 4 Langchain Definitions

class CustomRetrieverLang(BaseRetriever): 
    def get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # Perform the encoding and retrieval for a specific question
        ans = combined_pipe(query)
        ans = DataCollection(ans)
        answer=ans[0]['answer']
        answer_string = ' '.join(answer)
        return [Document(page_content=answer_string)]   
# Ensure correct VectorStoreRetriever usage
retriever = CustomRetrieverLang()

# Define the prompt template
template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Use three sentences maximum and keep the answer as concise as possible. 
Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""
rag_prompt = PromptTemplate.from_template(template)
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | rag_prompt
    | watsonx_granite
)

prompt = "I have started to get lots of acne on my face, particularly on my forehead what can I do"
prompt = "Can a painless tooth be extracted since it has a hole in it" 
# dataset['train'][14]
# {'Description': 'Q. Can a painless tooth be extracted since it has a hole in it?', 'Patient': 'Hello doctor,Can I pull out hard teeth out no pain because I got a hole in my teeth? Whenever I use to eat something, it got stuck in it and caused severe pain. Kindly give me some advice about pulling it out without any pain.', 'Doctor': 'Hello. I had gone through your question, and I can understand your concern. If the pain occurs only when you eat food and it gets stuck into it, it is reversible pulpitis. For that, you do not need to get it extracted. You need to get it filled, and it will cost you less than that of pulling (extraction). Second, you will save your tooth too. My suggestion is to get dental filling done as soon as possible. If you want to get it removed, you need to visit a doctor (dentist), and there will be no pain because during extraction local anesthesia is given.'}
if print_full_prompt:
    # Get the retrieved context
    context = retriever.get_relevant_documents(prompt)
    print("Retrieved context:")
    for doc in context:
        print(doc)
    # Construct the full prompt
    full_prompt = rag_prompt.format(context=context, question=prompt)
    print("Full prompt:", full_prompt)

print(rag_chain.invoke(prompt)) 
collection.load()
import towhee
def chat_function(message, history):
    history = history or []
    response = rag_chain.invoke(message)
    history.append((message, response))
    return history, history

import json
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    params = request.get_json()
    message = params.get('message')
    history = params.get('history', [])

    if not message:
        return jsonify({"message": "Missing 'message' parameter"}), 400

    history, chat_history = chat_function(message, history)

    response_data = {
        "message": message,
        "response": chat_history[-1][1],
        "history": chat_history
    }
    return jsonify(response_data), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)




