import os
from pathlib import Path
from decouple import config

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.output_parsers import StrOutputParser

from basic_chain import get_model
from rag_chain import make_rag_chain
from remote_loader import load_web_page
from splitter import split_documents
from vector_store import create_vector_db
from langchain.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings

OPENAI_API_KEY = config("OPENAI_API_KEY")
USE_HUGGINGFACE_EMBEDDINGS = False

DIR_VECTOR_DB = Path.cwd().joinpath("store", "chroma")
OPENAI_EMBEDDINGS = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small")

def ensemble_retriever_from_docs(docs, embeddings=None):
    texts = split_documents(docs)
    if DIR_VECTOR_DB.exists():
        vs = Chroma(persist_directory=str(DIR_VECTOR_DB), embedding_function=OPENAI_EMBEDDINGS)
    else:
        vs = create_vector_db(texts, embeddings)
    vs_retriever = vs.as_retriever()

    bm25_retriever = BM25Retriever.from_texts([t.page_content for t in texts])

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vs_retriever],
        weights=[0.5, 0.5])

    return ensemble_retriever

def main(): 
    problems_of_philosophy_by_russell = "https://www.gutenberg.org/ebooks/5827.html.images"
    docs = load_web_page(problems_of_philosophy_by_russell)
    ensemble_retriever = ensemble_retriever_from_docs(docs)
    model = get_model("ChatGPT")
    chain = make_rag_chain(model, ensemble_retriever) | StrOutputParser()

    result = chain.invoke("What are the key problems of philosophy according to Russell?")
    print(result)

if __name__ == "__main__":
    print(DIR_VECTOR_DB)
    quit()
    # this is to quite parallel tokenizers warning.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()

