from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter ,CharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate,ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

from langchain.vectorstores.utils import filter_complex_metadata
from langchain_community.llms import Ollama

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

class ChatPDF:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self):
        self.model = ChatOllama(model="llama3")
        #self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.text_splitter = CharacterTextSplitter(separator="\n",
                                        chunk_size=1000,
                                        chunk_overlap=200,
                                        length_function=len) 
        system_template = """
        "You are a helpful assistant. 한국어로 번역해서 답해주세요."""
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)


        # (b) Human Message
        human_template = """
                    <요구사항> 으로 시작하는 태그의 내용은 너가 지켜야 할 사항들이야. 이것을 지켜줘.
                    
                    <요구사항>
                    당신은 질문에 답변하는 AI봇입니다.
                    <참고>로 구분되는 텍스트를 기반으로 답변을 하세요.
                    답을 만들 수 없다면 모른다고 답해주세요.
                    <질문>으로 구분되는 텍스트가 질문입니다.
                    답변은 질문과 같은 언어를 사용하세요.
                    </요구사항>
                    
                    <참고>
                    {context}
                    </참고>
                    
                    
                    <질문> {question} </질문>
                    """
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

        self.prompt = ChatPromptTemplate.from_messages(
            [
                system_message_prompt,human_message_prompt
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

        '''
        context_text=self.retriever.invoke(query)
        prompt = self.prompt.format(context=context_text, question=query)
        return self.model.predict(prompt)
        '''
        return self.chain.invoke(query)

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None



#llm = Ollama( model = "mistral" )

#str = llm.invoke("Tell me a joke")

#print(str)