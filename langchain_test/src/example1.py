from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata
from langchain_community.llms import Ollama

from langchain_community.embeddings import OllamaEmbeddings


class ChatPDF:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self):
        self.model = ChatOllama(model="mistral")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST] 당신은 질문에 답변하는 답변자입니다. 참고내용을 사용하여 질문에 답하세요. 답을 모르면 모른다고 세 문장으로 표현하세요. 최대한 간결하게 답변하세요. [/INST] </s> 
            [INST] 질문: {question} 
            참고내용: {context} 
            답변: [/INST]
            """
        )
    
    def ingest(self, pdf_file_path: str):
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)

        ##vector_store = Chroma.from_documents(documents=chunks, embedding=FastEmbedEmbeddings())
        ollama_emb = OllamaEmbeddings(model="mistral",)
        vector_store = Chroma.from_documents(documents=chunks, embedding=ollama_emb)
        
        self.retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.5,
            },
        )

        self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
                      | self.prompt
                      | self.model
                      | StrOutputParser())

    def ask(self, query: str):
        if not self.chain:
            return "Please, add a PDF document first."

        return self.chain.invoke(query)

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None



llm = Ollama( model = "mistral" )

str = llm.invoke("Tell me a joke")

print(str)