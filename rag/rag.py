from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, WebBaseLoader, PyPDFLoader

load_dotenv()
llm1 = ChatGoogleGenerativeAI(model="gemini-2.0-flash")


######################################################################## Load from text file
loader = TextLoader("./speech.txt")
text_documents=loader.load()
# print(text_documents)



######################################################################## Load from web page
from bs4 import SoupStrainer
loader=WebBaseLoader(web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
                     bs_kwargs=dict(
                         parse_only=SoupStrainer(
                            class_=("post-title","post-content","post-header")
                     )))

text_documents=loader.load()
# print(text_documents)



######################################################################## Load from pdf file
loader = PyPDFLoader("./resume.pdf")
text_documents=loader.load()
# print(text_documents)



######################################################################## Splitting data into chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
chunks = splitter.split_documents(text_documents)
# print(chunks[:1])



######################################################################## Converting chunks to embeddings/vectors
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# texts_to_embed = []
# for chunk in chunks:
#     texts_to_embed.append(chunk.page_content)
# embedded_chunks = embedding_model.embed_documents(texts_to_embed)
embedded_chunks = embedding_model.embed_documents([chunk.page_content for chunk in chunks])



######################################################################## Store chunks in vector store
from langchain_community.vectorstores import FAISS
# Above step of converting chunks to embeddings manually doesn't need to be done if using .from_documents(), it handles that step for you
vectorDB = FAISS.from_documents(chunks, embedding_model)



######################################################################## Query and retrieve data
## OPTION 1
# chunks = vectorDB.similarity_search("what is the name of candidate?")
# for doc in chunks:
#     print(doc.page_content)

## OPTION 2
# retriever = vectorDB.as_retriever()
# from langchain.chains import RetrievalQA
# rag_chain = RetrievalQA.from_chain_type(
#     llm=llm1,
#     retriever=retriever,
#     return_source_documents=True
# )

# response = rag_chain.invoke("what is the name of candidate and where does the candidate work currently?")
# print(response["result"])



## OPTION 3
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context. 
Think step by step before providing a detailed answer. 
I will tip you $1000 if the user finds the answer helpful. 
<context>
{context}
</context>
Question: {input}""")

from langchain.chains.combine_documents import create_stuff_documents_chain
document_chain = create_stuff_documents_chain(llm1, prompt)
retriever = vectorDB.as_retriever()

from langchain.chains import create_retrieval_chain
rag_chain = create_retrieval_chain(retriever, document_chain)
query = "What is the name of the candidate and where does the candidate work currently?"
response = rag_chain.invoke({"input": query})

# print(response["answer"])





######################################################################## ADVANCED RAG PIPELINE WITH MULTIPLE DATA SOURCE







######################################################################## GROQ INFERENCE ENGINE
from langchain_groq import ChatGroq

llmGroq = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.0,
    max_retries=2,
    # other params...
    )
promptTemplate = ChatPromptTemplate.from_messages([
    ("system", "You're a pro sentiment analyzer."),
    ("human", "{text}")
])

prompt = promptTemplate.invoke({"text": "i am really happy today. Good mood"})
ai_msg = llmGroq.invoke(prompt)
print(ai_msg)

