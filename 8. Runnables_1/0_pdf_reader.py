from langchain_openai import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSpliiter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import os 

load_dotenv()
# load the document
loader = TextLoader("docs.txt")
documents = loader.load()

# split the text into smaller chunks
text_splitter = RecursiveCharacterTextSpliiter(chunk_size = 500, chunk_overlap = 50)
docs = text_splitter.split_documents(documents)

# convert the into embeddings and store in FAISS
vectorstore = FAISS.from_documents(docs,OpenAIEmbeddings())

# create a retriever
retriever = vectorstore.as_retriever()

# Manually retrived relevant docyments
query = "what are the takeaways from the document?"
retrieved_docs = retriever.get_relevent_documents(query)

# Combine Retrieved text into a Single Prompt
retrieved_text = "\n".join([doc.page_content for doc in retrieved_docs])

#initialize the model
model = ChatOpenAI( model = os.getenv("DEEPSEEK_MODEL"))

# Manually pass Retrived text to Model
prompt = f"Based on the following text , answer the questions : {query}\n\n{retrieved_text}"
answer = model.invoke(prompt)

print("Answer:",answer)