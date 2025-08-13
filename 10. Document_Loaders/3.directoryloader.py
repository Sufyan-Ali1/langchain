from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader

loader = DirectoryLoader(
    path = r"C:\Users\sufya\Desktop\Learning\Langchain\10. Document_Loaders\3.directory",
    glob = "*.pdf",
    loader_cls = PyPDFLoader
    )



docs = loader.load()

print(len(docs))
print(docs[0])