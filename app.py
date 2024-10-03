from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma 
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter


######### Content Processing Block ###############################

## Loading PDF file from local file directory
## read the content and store it in data object 
local_path = "VIOV-Vanguard S&P Small-Cap 600 Value ETF _ Vanguard.pdf"

if local_path:
    loader = UnstructuredPDFLoader(file_path=local_path)
    data = loader.load()
else:
    print("Upload a PDF file for processing.")

print(data[0].page_content)


## Converting content into dense vector embeddings 
#Split and chunk the data
text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
chunks = text_splitter.split_documents(data)


# Add the chunks to vector database, which takes the model for creating the embeddings.
vector_db = Chroma.from_documents(
    documents=chunks, 
    embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True),
    collection_name="local-rag"
)
###################################################

######### Retrieval + Generation of Response ##############################
local_llm = "llama3.1"
llm = ChatOllama(model = local_llm)

QUERY_PROMPT = PromptTemplate(
    input_variables = ["question"],
    template="""You are a consultant specializing in investment funds for a large company. Your task is to generate five different versions of the given user question to retrieve relavant documents about funds from a vector databaase. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search. Provide these alternative questions separated by newlines. 
    Original question: {question} """
)

retriever = MultiQueryRetriever.from_llm(vector_db.as_retriever(), llm, prompt=QUERY_PROMPT)

# RAG Prompt
template = """Answer the question based ONLY on the following context: 
{context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {
        "context":retriever, 
        "question": RunnablePassthrough()
    }
    | prompt 
    | llm 
    | StrOutputParser()
)
q = "Give me a ricks evaluation about the fund"
response = chain.invoke(q)

print(response)

###################################################