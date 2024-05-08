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
        self.model = ChatOllama(model='llama3-ko-8b-q4')
        
        ##### === Splitter 셋팅
        #self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.text_splitter = CharacterTextSplitter(separator="\n",
                                        chunk_size=200,
                                        chunk_overlap=100,
                                        length_function=len) 
        
        ##### === Prompt 셋팅
        ### 기본 셋팅
        system_template = """
        당신은 질문에 답변하는 한국어 AI봇입니다.
        """
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

        human_template = r"""
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
                system_message_prompt, human_message_prompt
            ]
        )
        
        ### 메모리 테스트용 template
        system_template = r"""
        당신은 질문에 답변하는 한국어 AI봇입니다. <학습>으로 시작하고 </학습>으로 끝나는 내용을 기반으로 답하세요.
        """
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

        human_template = """
                    {question} 
                    """
                    
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

        self.prompt = ChatPromptTemplate.from_messages(
            [
                system_message_prompt,MessagesPlaceholder(variable_name="chat_history"), human_message_prompt
            ]
        )
        
        # 메모리 셋팅
        self.memory = ConversationSummaryBufferMemory(
            llm=self.model,
            max_token_limit=80,
            memory_key="chat_history",
            return_messages=True,
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
        ### 메모리
        handler_2 = StdOutCallbackHandler()
        self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
                      | RunnablePassthrough.assign(chat_history=self.load_memory)
                      | self.prompt
                      | self.model
                      
                      | StrOutputParser()
                      )     
        
        ### 미리 교육
        self.preLearning(chunks)   
    
     ### 메모리
    def load_memory(self,input):
        print(input)
        return self.memory.load_memory_variables({})["chat_history"]    
    
    def preLearning(self,chunks):
        q = "너는 <학습> 으로 시작하고 </학습> 으로 끝날 때 까지 해당 내용을 학습해야해."
        result= self.chain.invoke(q)
        self.memory.save_context(
                        {"input": q},
                        {"output": result},
                    )
        q = "<학습>"
        self.chain.invoke(q)
        self.memory.save_context(
                        {"input": q},
                        {"output": result},
                    )
        
        for _s in chunks:
            print("\n문장-----------------\n")
            print(_s.page_content)
            if _s:
                result=self.chain.invoke(_s.page_content)
                self.memory.save_context(
                        {"input": _s.page_content},
                        {"output": result},
                    )
                
        q = "<학습>"
        self.chain.invoke(q)
        self.memory.save_context(
                        {"input": q},
                        {"output": result},
                    )
        
        pass

    def ask(self, query: str):
        if not self.chain:
            return "Please, add a PDF document first."

        '''
        context_text=self.retriever.invoke(query)
        prompt = self.prompt.format(context=context_text, question=query)
        return self.model.predict(prompt)
        '''
        result=self.chain.invoke(query)
        self.memory.save_context(
                        {"input": query},
                        {"output": result},
                    )
        
        return result

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None



#llm = Ollama( model = "mistral" )

#str = llm.invoke("Tell me a joke")

#print(str)