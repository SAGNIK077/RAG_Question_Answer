import os

from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA

load_dotenv()

#working file directory like document directory. we used absolute directory instead of hard coded
working_dir = os.path.dirname(os.path.abspath(__file__))

#load embedding model
embedding = HuggingFaceEmbeddings()

#load llm
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.0
)


#read the pdf and store the value to vector DB. Parameter is file name

def process_documents_to_chroma_db(file_name):
    # load the file and read the file using unstructured module
    loader = UnstructuredPDFLoader(f"{working_dir}/{file_name}")
    documents = loader.load()

    #split documents to chunks
    #initialte the chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )

    #convert docs to chunks
    texts = text_splitter.split_documents(documents)

    #store chunks to vectordb/chromadb

    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embedding,
        persist_directory=f"{working_dir}/doc_vectorstores"
    )
    return 0



# answer the user question where input is user question

def answer_question(user_question):
    # load the persistent vector db means chroma db
    vectordb = Chroma(
        persist_directory=f"{working_dir}/doc_vectorstores",
        embedding_function=embedding
    )

    # create a retriever for document search
    retriever = vectordb.as_retriever()

    # get the response from the llm model
    qa_chain = RetrievalQA.from_chain_type(
        retriever=retriever,
        llm=llm,
        chain_type="stuff"
    )

    response = qa_chain.invoke({"query":user_question})
    answer = response["result"]

    return answer

