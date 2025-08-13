from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(r"C:\Users\sufya\Desktop\Learning\Langchain\10. Document_Loaders\5.csv")

docs = loader.load()

print(len(docs))
print(docs[0])