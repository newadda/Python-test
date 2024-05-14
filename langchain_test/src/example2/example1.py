from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter ,CharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate,ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder




from langchain.vectorstores.utils import filter_complex_metadata
from langchain_community.llms import Ollama

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings


####=========== 메모리
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationSummaryBufferMemory

####=========== 로그
from langchain_core.callbacks import FileCallbackHandler, StdOutCallbackHandler
from langchain.globals import set_verbose
set_verbose(True)

class ChatPDF:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self):
        
        ##### === Model 셋팅
        self.model = ChatOllama(model='ggml-model-q4_1')
        
        ##### === Splitter 셋팅
        #self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.text_splitter = CharacterTextSplitter(separator="\n",
                                        chunk_size=200,
                                        chunk_overlap=100,
                                        length_function=len) 
        
        ##### === Prompt 셋팅
        ### 기본 셋팅
        system_template = """
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. 모든 대답은 한국어(Korean)으로 대답해줘.          
"""
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

        human_template = """
Please comply with the requirements.
        
                    
Requirements:
Answer the questions based on the notes.
Keep your answers concise.
If you can't come up with an answer, don't force yourself to answer and just say you don't know.
Use the same language as the question in your answer.


reference:
{context}
                    
                    
question:
{question}
                   
"""
                    

        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

        self.prompt = ChatPromptTemplate.from_messages(
            [
                system_message_prompt, human_message_prompt
            ]
        )
        
        
    def ingest(self, pdf_file_path: str):
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)
        
        
        #---------------------------------------------------------------
        hf = HuggingFaceEmbeddings(model_name='jhgan/ko-sroberta-multitask')
        vector_store = Chroma.from_documents(documents=chunks, embedding=hf)
        #---------------------------------------------------------------
        #vector_store = Chroma.from_documents(documents=chunks, embedding=FastEmbedEmbeddings())
        #--------------
        ##ollama_emb = OllamaEmbeddings(model="llama3",)
        ##vector_store = Chroma.from_documents(documents=chunks, embedding=ollama_emb)
        #-------------
        self.retriever = vector_store.as_retriever(
           # search_type="similarity_score_threshold",
           # search_kwargs={
           #     "k": 3,
           #     "score_threshold": 0.9,
           # },
        )

        self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
                      | self.prompt
                      | self.model
                      | StrOutputParser())
      


   

    def ask(self, query: str):
        if not self.chain:
            return "Please, add a PDF document first."


        result=self.chain.invoke(query)

        
        return result

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None
