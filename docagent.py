import json
import os
import re
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import subprocess
from llama_cpp import Llama

class DocAgent:
    def __init__(
            self,
            modelPath,
            model_kwargs,
            encode_kwargs,
            llm_path
    ) -> None:
        self.embeddings = HuggingFaceEmbeddings(
            model_name=modelPath,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        gpu = 0
        try:
            subprocess.check_output('nvidia-smi')
            print('Nvidia GPU detected!')
            gpu = 1
        except Exception: # this command not being found can raise quite a few different errors depending on the configuration
            print('No Nvidia GPU in system!')

        self.llm = Llama(
            model_path=llm_path,
            n_ctx=4096,
            n_threads=os.cpu_count()-2,
            n_gpu_layers=40 if gpu else 0,
            chat_format='chatml',
            n_batch=768 if gpu else 512,
        )

        self.messages = []
        self.messages.append(
    {
        "role": "system",
        "content": """You are DocGPT, a chatbot that helps users to interact with documents. Be consice and clear in your answers. If you don't understand the question, you can ask for clarification. Don't hallucinate.
You have access to the document uploaded by the user. Simply call the function "query_document" with the search_query parameter to get the relevent contents from the document.
- query_document: Get the answer to a question from a given document. It'll return the most relevant content from the document. Always use this function if the user is asking about the document content or related to that.
    - parameters:
        - search_query: string (required) - Use keywords to search the document. 

If you need to use function, Use following format to respond. Make sure the argument in the function call tag can be parsed as a JSON object.
<functioncall>{"search_query": "value"}</functioncall>

If you don't want to use the function, just don't include any function call tags in the response.
NOTE:
- User'll ask question regarding to the document or book or any other doc content. You need to answer the question by calling the function "query_document" with the search_query parameter. You are access to the document they uploaded.
- Initiall there is a document uploaded to you regards 'way to build a successful web application'."""
    }
)
        self.db = None
        self.create_db("./documents")

    def check_for_function_call(self, req):
        if "<functioncall>" in req and "</functioncall>" in req:
            reg = re.compile(r'<functioncall>(.*?)</functioncall>', re.DOTALL)
            match = reg.search(req)
            fn_call = match.group(1)
            return fn_call
        return None
    
    def create_db(self, path):
        loader = PyPDFDirectoryLoader(path=path)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = text_splitter.split_documents(data)
        db = FAISS.from_documents(docs, self.embeddings)
        self.db = db.as_retriever()
        print("Database created successfully!")

    def get_response(self, query):
        self.messages.append(
            {
                "role": "user",
                "content": query
            }
        )
        llm_response = self.llm.create_chat_completion(
            messages=self.messages,
            stream=False,
        )["choices"][0]["message"]["content"]
        fn_call = self.check_for_function_call(llm_response)
        if fn_call is not None:
            print("Function call found: ", fn_call)
            fn_args = json.loads(fn_call)
            if self.db is not None:
                res = self.db.get_relevant_documents(fn_args["search_query"])
            else:
                res = "No documents uploaded."
            print("query response: ", res)
            self.messages.append(
                {
                    "role": "user",
                    "content": "This is the function call response (NOT USER): " + str(res) + "Take this to user and answer the question based on it."
                }
            )
            llm_response = self.llm.create_chat_completion(
                messages=self.messages,
                stream=False,
            )["choices"][0]["message"]["content"]
            return llm_response
        else:
            return llm_response

                
            