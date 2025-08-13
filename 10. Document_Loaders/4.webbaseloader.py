from langchain_community.document_loaders import WebBaseLoader

url = "https://www.daraz.pk/products/shan-biryani-master-bundle-i878191169-s3886049304.html?scm=1007.51610.379274.0&pvid=669db145-73db-42e3-b374-e78bcb1b2aad&search=flashsale&spm=a2a0e.tm80335142.FlashSale.d_878191169"

loader = WebBaseLoader(url)# you can pass here list of urls 

docs = loader.load()

print(docs[0].page_content)