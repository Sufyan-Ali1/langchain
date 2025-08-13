from langchain_community.document_loaders import TextLoader

loader = TextLoader(r"C:\Users\sufya\Desktop\Learning\Langchain\10. Document_Loaders\1.txt")

docs = loader.load()

print(docs[0])