# Standard library import
import logging
from decouple import config
import os
import random
import time
import json
import boto3

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
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ChatMessageHistory
from langchain.chains import LLMChain
import openai


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
ACCOUNT_SID = config("TWILIO_ACCOUNT_SID")
AUTH_TOKEN = config("TWILIO_AUTH_TOKEN")
OPENAI_API_KEY = config("OPENAI_API_KEY")
TWILIO_NUMBER = config('TWILIO_NUMBER')
USE_TEXT_FILES = True
USE_HUGGINGFACE_EMBEDDINGS = False
FILE_TRANSLATION_EXAMPLES = "./translation_examples.json"

# Setup S3 bucket connection
AWS_ACCESS_KEY_ID = config("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = config("AWS_SECRET_ACCESS_KEY")

SESSION = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
S3 = SESSION.client('s3')
BUCKET_NAME = "chichewa-ai"
USERS_FILE_KEY = 'phoso-ai-files/welcomed_users.json'

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_welcomed_users():
    """
    Loads the list of welcomed users from an S3 bucket.

    This function attempts to download and load a JSON file from an S3 bucket 
    that contains the list of phone numbers for users who have already received 
    a welcome message. If the file does not exist, the function returns an empty 
    dictionary.

    Returns
    -------
    dict
        A dictionary where the keys are user phone numbers and the values indicate 
        whether the user has been welcomed. If the S3 file does not exist, an 
        empty dictionary is returned.

    Raises
    ------
    botocore.exceptions.BotoCoreError
        If there is an error in accessing the S3 bucket, such as network issues 
        or incorrect credentials.
    """
    try:
        # Download the file from S3
        s3_response = S3.get_object(Bucket=BUCKET_NAME, Key=USERS_FILE_KEY)
        users_data = s3_response['Body'].read().decode('utf-8')
        return json.loads(users_data)
    except S3.exceptions.NoSuchKey:
        # If the file doesn't exist, return an empty dictionary
        return {}

def save_welcomed_users(welcomed_users):
    """
    Saves the list of welcomed users to an S3 bucket.

    This function takes a dictionary of welcomed users, converts it into a JSON string,
    and uploads it to a specified S3 bucket. The JSON file stores the phone numbers of users 
    who have already received a welcome message.

    Parameters
    ----------
    welcomed_users : dict
        A dictionary where the keys are user phone numbers and the values indicate 
        whether the user has been welcomed.

    Raises
    ------
    botocore.exceptions.BotoCoreError
        If there is an error in uploading the file to the S3 bucket, such as network issues 
        or incorrect credentials.
    """
    # Convert the dictionary to a JSON string
    users_data = json.dumps(welcomed_users)
    # Upload the JSON string to S3
    S3.put_object(Bucket=BUCKET_NAME, Key=USERS_FILE_KEY, Body=users_data)

def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = (openai.RateLimitError,),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            # Retry on specified errors
            except errors as e:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper

def load_translation_examples(file_path, source_language, target_language):
    """
    Loads and formats translation examples from a JSON file.

    Parameters
    ----------
    file_path : str
        The path to the JSON file containing translation examples.
    source_language : str
        The source language (e.g., "Chichewa").
    target_language : str
        The target language (e.g., "English").

    Returns
    -------
    list
        A list of formatted translation examples.
    """
    with open(file_path, 'r') as file:
        examples = json.load(file)
    
    key = f"{source_language}-{target_language}"
    if key in examples:
        return examples[key]
    else:
        raise ValueError(f"Translation examples for {source_language} to {target_language} not found.")

def translate_text_openai(text, source_language, target_language, llm=None):
    """
    Translates the given text from the source language to the target language using an LLM with few-shot examples.

    Parameters
    ----------
    text : str
        The text to be translated.
    source_language : str
        The source language of the text.
    target_language : str
        The language into which the text should be translated.
    examples : list
        A list of few-shot translation examples.

    Returns
    -------
    str
        The translated text.
    """
    # Load examples 
    examples = load_translation_examples(FILE_TRANSLATION_EXAMPLES, 
                                         source_language, target_language)

    # Construct the prompt template using examples
    example_prompts = "\n".join([f'{source_language}: "{ex[source_language]}"\n{target_language}: "{ex[target_language]}"' for ex in examples])

    prompt_template = PromptTemplate.from_template(
        f"""
        You are a professional translator who specializes in translating text from {source_language} to {target_language}.
        Given the following examples, translate the provided text.

        Examples:
        {example_prompts}

        Now, translate the following:

        {source_language}: "{{text}}"
        {target_language}:
        """
    )

    # Initialize the chat-based model
    if not llm:
        llm = ChatOpenAI(temperature=0.7, model="gpt-4o", 
                         openai_api_key=OPENAI_API_KEY)

    # Create the LLMChain for translation
    translation_chain = LLMChain(llm=llm, prompt=prompt_template)

    # Perform the translation
    return translation_chain.run({"text": text})

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

def detect_language_with_langchain(text, llm=None):
    
    language_detection_prompt = PromptTemplate.from_template(
    """
    You are a language detection expert. Your task is to identify the language of the given text accurately.
    Respond with only the name of the language (e.g., "English", "Chichewa", "Spanish", etc.).

    Here are some examples:
    
    Example 1:
    Text: "Hello, how are you?"
    Language: English
    
    Example 2:
    Text: "Moni, muli bwanji?"
    Language: Chichewa
    
    Example 3:
    Text: "ndikuti kukupezeka nyemba zambiri"
    Language: Chichewa
    
    Now, identify the language for the following text:
    
    Text: "{text_to_detect}"
    Language:
    """)

    if not llm:
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY)

    # Create the LLMChain with the prompt
    language_detection_chain = language_detection_prompt | llm | StrOutputParser()

    # Detect the language
    detected_language = language_detection_chain.invoke({"text_to_detect": text})

    return detected_language

def detect_language_with_transformers(text):
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