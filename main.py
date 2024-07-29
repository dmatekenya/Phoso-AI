# Third-party imports
from langdetect import detect
from langchain_openai import OpenAI, ChatOpenAI
from langchain.chains import LLMChain
from fastapi import FastAPI, Form, Depends, Request, HTTPException, BackgroundTasks
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from twilio.rest import Client
from twilio.http.http_client import TwilioHttpClient
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

from twilio.twiml.messaging_response import MessagingResponse
from fastapi.responses import PlainTextResponse
from decouple import config
from datetime import datetime
import time 
import json

# Find your Account SID and Auth Token at twilio.com/console
# and set the environment variables. See http://twil.io/secure
ACCOUNT_SID = config("TWILIO_ACCOUNT_SID")
AUTH_TOKEN = config("TWILIO_AUTH_TOKEN")
OPENAI_API_KEY = config("OPENAI_API_KEY")
FILE_TRANSLATION_EXAMPLES = "./translation_examples.json"

# Initialize a custom HTTP client with a timeout
http_client = TwilioHttpClient()
http_client.session.timeout = 60  # Set timeout to 60 seconds
CLIENT = Client(ACCOUNT_SID, AUTH_TOKEN,http_client=http_client)
TWILIO_NUMBER = config('TWILIO_NUMBER')

# Internal imports
from models import Conversation, SessionLocal
from utils import logger, run_rag_query, detect_language
from sql_chain import run_sql_chain
import logging

app = FastAPI()

def format_translation_examples(examples_file, source_language, target_language):
    examples = load_examples(examples_file)
    key = f"{source_language}-{target_language}"
    if key in examples:
        return "\n".join([f"{source_language}: {ex[source_language]}\n{target_language}: {ex[target_language]}" 
                          for ex in examples[key]])
    return ""

def load_examples(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# Prepare Translation Variables
TRANSLATION_EXAMPLES = load_examples(FILE_TRANSLATION_EXAMPLES)

CHAT_MODEL = ChatOpenAI(temperature=0.7, openai_api_key=OPENAI_API_KEY, 
                            model="gpt-3.5-turbo")
# Create a system message with examples
TRANSLATION_SYSTEM_TEMPLATE = """You are a professional translator. Your task is to translate {source_language} to {target_language}.
    Here are a few examples:

    {examples}

    Now, translate the following text:"""

TRANSLATION_SYSTEM_MESSAGE_PROMPT = SystemMessagePromptTemplate.from_template(TRANSLATION_SYSTEM_TEMPLATE)


def translate_text_openai(text, source_language, target_language):
    # Create a human message for the actual translation request
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    # Combine the prompts
    chat_prompt = ChatPromptTemplate.from_messages([TRANSLATION_SYSTEM_MESSAGE_PROMPT, human_message_prompt])

    # Create an LLMChain for translation
    translation_chain = LLMChain(llm=CHAT_MODEL, prompt=chat_prompt)
    
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

# Dependency
def get_db():
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()

def send_whatsapp_message(body, to):
    try:
        message = CLIENT.messages.create(
            body=body,
            from_=TWILIO_NUMBER,
            to=to
        )
        print(f"Message sent successfully: SID {message.sid}")
    except Exception as e:
        print(f"Failed to send message: {e}")

def simple_router(incoming_msg):
    commodities_price = ['Maize', 'Rice', 'Soya beans', 'Beans', 'Cow peas', 'Groundnuts']
    crop_estimates = ['Maize', 'Beans', 'Cow peas', 'Dolichus beans ', 'Soy beans',
       'Ground beans', 'Paprika', 'Rice', 'Pigeon peas', 'Grams',
       'Sesame ', 'Field peas', 'Velvet beans', 'Chick peas', 'Wheat',
       'Millet', 'Sorghum ', 'Groundnuts', 'Cassava', 'Sweet potatoes',
       'Potatoes', 'Tobacco', 'Flue cured', 'Sunflower ', 'Chillies',
       'Cotton ', 'Bananas', 'Mangoes', 'Oranges', 'Tangerines', 'Coffee',
       'Pineapples', 'Guava', 'Pawpaws', 'Peaches', 'Lemons',
       'Grape fruits', 'Apples', 'Avocado pear', 'Macademia', 'Tomatoes',
       'Onions', 'Cabbage', 'Egg plants', 'Okra', 'Cucumber']
    price_estimates_key_words = ["price", "cheap", "produce", 
                                 "buy", "sell", "sale", "find"]
    all_kw = [i.lower() for i in set(commodities_price+crop_estimates+price_estimates_key_words)]

    use_sql = False
    
    for word in incoming_msg.lower().split():
        if word in all_kw:
            use_sql = True
            return use_sql
    return use_sql
    
def handle_incoming_message(original_incoming_msg, sender):
    
    print("Question=", original_incoming_msg)
    # Implement your chatbot logic here
    msg_lan = detect_language(original_incoming_msg)
    print(msg_lan)
    if msg_lan != "en":
        incoming_msg = translate_text_openai(original_incoming_msg, "Chichewa", "English")
        print("Translated Question=", incoming_msg)
    else:
        incoming_msg = original_incoming_msg
    
    use_sql = simple_router(incoming_msg)
    if use_sql:
        response_text = run_sql_chain(incoming_msg, lan=msg_lan)
    else:
        response_text = run_rag_query(incoming_msg)
    
    # Translate text to Chichewa if incoming message was in Chichewa
    # print("English Response=>", response_text)
    # if msg_lan != "en":
    #     response_text = translate_text_openai(response_text, "English", "Chichewa")
    #     print("Translated Response=>", response_text)
    
    # print("=======================================")
    # print("FINAL MESSAGE")
    # print("=======================================")
    # print(response_text)
    # print("-"*50)

    # end = datetime.now()
    # time_taken = (end - start).total_seconds()
    # print("This took {} seconds".format(time_taken))
    # print("-"*50)
    
    # Send a response using the Twilio Client
    send_whatsapp_message(response_text, sender)
    return response_text

def process_user_request(incoming_msg, sender):
    try:
        # Handle the incoming message and generate the response
        response_text = run_sql_chain(incoming_msg)
        # Send the response using Twilio API
        send_whatsapp_message(response_text, sender)
    except Exception as e:
        print(f"Error in background task: {e}")


@app.post("/message")
async def sms_reply(request: Request, Body: str = Form(...), From: str = Form(...)):
    try:
        incoming_msg = Body
        whatsapp_number = From

        # Handle the incoming message and send a response
        response_text = handle_incoming_message(incoming_msg, whatsapp_number)

        return PlainTextResponse(response_text)
    except Exception as e:
        logger.error(f"Error sending message to {whatsapp_number}: {e}")
        raise HTTPException(status_code=500, detail="Error processing message")


@app.post("/fallback")
async def fallback(request: Request, Body: str = Form(...), From: str = Form(...)):
    try:
        # Message to send when there is a problem
        detected_lan = detect_language(Body)
        if detected_lan == "sw" or detected_lan  == "ny":
            fallback_message = "Pepani, koma sindingathe kuyankha funso lanu panopa chifukwa chabvuto lina. Kodi pali funso lina lomwe mulinalo?"
        else:
            fallback_message = "I'm sorry, but I can't generate an answer for your query right now. Is there anything else I can assist you with?"
    
        send_whatsapp_message(fallback_message, From)
        return PlainTextResponse(fallback_message)
    except Exception as e:
        print(f"Error handling /fallback endpoint: {e}")
        raise HTTPException(status_code=500, detail="Error processing fallback message")



