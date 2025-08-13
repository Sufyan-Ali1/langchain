from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(r"C:\Users\sufya\Desktop\Learning\Langchain\10. Document_Loaders\2.pdf")

docs = loader.load()

print(len(docs))
print(docs[0])