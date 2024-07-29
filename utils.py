# Standard library import
import logging
from decouple import config
import os
import json

# Third-party imports
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
from fastapi.responses import PlainTextResponse

from transformers import pipeline
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import OpenAI, ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ChatMessageHistory
from langchain.chains import LLMChain


# Local imports
from ensemble import ensemble_retriever_from_docs
from rag_chain import make_rag_chain, get_question
from local_loader import load_pdf_files, load_txt_files
from basic_chain import basic_chain, get_model
from splitter import split_documents
from vector_store import create_vector_db
from memory import create_memory_chain


# Find your Account SID and Auth Token at twilio.com/console
# and set the environment variables. See http://twil.io/secure
ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TWILIO_NUMBER = os.getenv('TWILIO_NUMBER')
USE_TEXT_FILES = True
USE_HUGGINGFACE_EMBEDDINGS = False
FILE_TRANSLATION_EXAMPLES = "./translation_examples.json"


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_examples(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def format_translation_examples(examples_file, source_language, target_language):
    examples = load_examples(examples_file)
    key = f"{source_language}-{target_language}"
    if key in examples:
        return "\n".join([f"{source_language}: {ex[source_language]}\n{target_language}: {ex[target_language]}" 
                          for ex in examples[key]])
    return ""

def translate_text_openai(text, source_language, target_language):
    # Function to translate text
    formated_examples = format_translation_examples(FILE_TRANSLATION_EXAMPLES, 
                                                    source_language, 
                                    target_language)
    return translation_chain.run({
            "source_language": source_language,
            "target_language": target_language,
            "examples": formated_examples,
            "text": text
        })


def get_retriever():
    if USE_TEXT_FILES:
        docs = load_txt_files()
    else:
        docs = load_pdf_files()
    
    if USE_HUGGINGFACE_EMBEDDINGS:
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    else:
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small")
    
    return ensemble_retriever_from_docs(docs, embeddings=embeddings)

def get_chain():
    model = get_model("ChatGPT")
    chat_memory = ChatMessageHistory()
    ensemble_retriever = get_retriever()
    output_parser = StrOutputParser()
    rag_chain = make_rag_chain(model, ensemble_retriever) 
    chain = create_memory_chain(model, rag_chain, chat_memory) | output_parser
    
    return chain

def run_rag_query(query):
    """Helper function to run RAG Query

    """
    memory_chain = get_chain()
    response = memory_chain.invoke(
            {"question": query},
            config={"configurable": {"session_id": "foo"}}
        )
    return response

     
def route_query_en(query):
     price_and_quantity_keywords = ["how much", "price", 
                                    "cheap", "expensive", "maize", 
                                    "rice", "beans", "tobaccon", 
                                    "produced", "harvest", "harvested"]
     pass
     

def detect_language(text):
    # Initialize the language identification pipeline
    lang_id = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")

    # Detect language
    result = lang_id(text)
    return result[0]['label']
     

def main():
    # questions = [
    #     "Combien de personnes sont mortes du paludisme au Cameroun ??",
    #     "Combien de personnes disposent de moustiquaires au Cameroun?"]
    # for q in questions:
    #     print("\n--- QUESTION: ", q)
    #     output = run_rag_query(q)
    #     print('OUTPUT TYPE==>', output)
    #     #print("* RAG:\n", chain.invoke(q))
    
    # Example usage
    source_texts = ["Mtedza ndingaupeze kuti?", 
                    "Mpunga ukugulitsidwa pa mtengo wanji?",
                    "Chimanga chotchipa chikupezeka kuti?",
                    "Ndikuti kunakololedwa mtedza wambiri?"]
    for text in source_texts:
        print("\n--- QUESTION: ", text)
        translated_text = translate_text_openai(text, "Chichewa", "English")
        print(translated_text)


if __name__ == '__main__':
    # this is to quite parallel tokenizers warning.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()