# Third-party imports
import os
from decouple import config
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from fastapi import FastAPI, Form, Depends, Request, HTTPException, BackgroundTasks
from twilio.rest import Client
from twilio.http.http_client import TwilioHttpClient
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

from twilio.twiml.messaging_response import MessagingResponse
from fastapi.responses import PlainTextResponse, Response
from datetime import datetime
import json
import asyncio

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
from utils import *
from sql_chain import process_sql_query, run_sql_chain
import logging

# Load the users when the app starts
WELCOMED_USERS = load_welcomed_users()
print("WELCOMED_USERS==>", WELCOMED_USERS)

# Initialize app
app = FastAPI()

def generate_welcome_message(user_message, llm=None):
    
    # Define the prompt template for the welcome message
    welcome_message_prompt = PromptTemplate.from_template(
    """
    You are a helpful assistant who interacts with users in their preferred language. 
    Your task is to greet the user in their language and briefly inform them about the type of information you can provide.

    Specifically, you provide the following types of information:
    - Prices for food and other agricultural commodities (e.g., Maize, rice, soy beans).
    - Agricultural production details (e.g., Maize, Tobacco).
    - The situation of food security (e.g., how many people are lacking food).
    - All information is specific to Malawi.

    Greet the user and let them know that they can ask questions about these topics. 
    
    Here are some examples:
    
    Example 1:
    Text: "Hello, how are you?"
    Welcome Message: "Welcome! I can help you with information about food prices, agricultural production, and food security in Malawi. For example, you can ask: 'What is the current price of maize?' or 'How much maize was produced last year?How can I assist you today?'"
    
    Example 2:
    Text: "Moni, muli bwanji?"
    Welcome Message: "Takulandirani! Ndikhoza kukuthandizani ndi zambiri zokhudzana ndi mitengo ya chakudya, zokolola, zaulimi komanso zokhudzana ndi zanjala mmene ilili ku Malawi. Mwachitsanzo mutha kufunsa kuti: "Kodi chimanga chili pa bwanji ku Kasungu?"
    'Kodi ndikuti kunakololedwa mtedza wambiri?' kapena 'Kodi ndikuti kunakololedwa mtedza wambiri?'. Kodi ndingakuthandizeni bwanji lero?"
    

    Now, based on the user's input, generate a suitable welcome message in the same language.

    Text: "{text_to_detect}"
    Welcome Message:
    Example Questions:
    1. 
    2. 
    """)

    # Initialize the chat-based model (e.g., GPT-3.5-turbo)
    if not llm:
        llm = ChatOpenAI(temperature=0.7)

    # Create the LLMChain with the prompt
    welcome_message_chain = welcome_message_prompt | llm | StrOutputParser()

    return welcome_message_chain.invoke({"text_to_detect": user_message})

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
    
def handle_incoming_message(original_incoming_msg, user_number, sql_only=True):
    # Send dummy message
    send_whatsapp_message("TESTING SENDING MESSAGE....", user_number)
    
    # Check if the user has already been welcomed
    if user_number not in WELCOMED_USERS:
        # Send a welcome message
        welcome_message = generate_welcome_message(user_message=original_incoming_msg)
        print("WELCOME MESSAGE==>", welcome_message)
        send_whatsapp_message(body=welcome_message, to=user_number)
        
        # Mark this user as welcomed
        WELCOMED_USERS[user_number] = True
        
        # Save the updated list of welcomed users to S3
        save_welcomed_users(WELCOMED_USERS)

        # Respond to their question
        if sql_only:
            print("Question=>", original_incoming_msg)
            response_text = process_sql_query(original_incoming_msg)
            print("LLM-response=>", original_incoming_msg)
            # Send a response using the Twilio Client
            send_whatsapp_message(response_text, user_number)
            return response_text
        else:
            pass
    else:
        # Respond to the user's message
        if sql_only:
            print("Question=>", original_incoming_msg)
            response_text = process_sql_query(original_incoming_msg)
            print("LLM-response=>", response_text)
            # Send a response using the Twilio Client
            send_whatsapp_message(response_text, user_number)
            return response_text
        else:
            pass
    # print("Question=", original_incoming_msg)
    # # Implement your chatbot logic here
    # msg_lan = detect_language(original_incoming_msg)
    
    # if msg_lan != "en":
    #     incoming_msg = translate_text_openai(original_incoming_msg, "Chichewa", "English")
    #     print("Translated Question=", incoming_msg)
    # else:
    #     incoming_msg = original_incoming_msg
    
    # use_sql = simple_router(incoming_msg)
    
    # if use_sql:
    #     response_text = run_sql_chain(incoming_msg, lan=msg_lan)
    # else:
    #     print("Checkin if we are going into RAG .....")
    #     response_text = run_rag_query(incoming_msg)
    
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

        # Create a Twilio MessagingResponse object
        response = MessagingResponse()

        # Handle the incoming message and generate a response
        response_text = process_sql_query(incoming_msg)
        # response_text = handle_incoming_message(incoming_msg, whatsapp_number)
        response.message(response_text)

        # Check if the user has already been welcomed
        if  whatsapp_number not in WELCOMED_USERS:
            print("In new user code .....")
            # Send a welcome message
            welcome_message = generate_welcome_message(user_message=incoming_msg)
            
            # First message: Welcome or acknowledgment
            response.message(welcome_message)

            # Mark this user as welcomed
            WELCOMED_USERS[ whatsapp_number] = True
            
            # Save the updated list of welcomed users to S3
            save_welcomed_users(WELCOMED_USERS)
        
        # Return the Twilio response as XML
        return Response(content=str(response), media_type="application/xml")
    except Exception as e:
        logger.error(f"Error sending messages to {whatsapp_number}: {e}")
        raise HTTPException(status_code=500, detail="Error processing message")

@app.post("/fallback")
async def fallback(request: Request, Body: str = Form(...), From: str = Form(...)):
    try:
        # Message to send when there is a problem
        detected_lan = detect_language_with_langchain(text=Body, llm=None)
        if detected_lan == "sw" or detected_lan  == "ny":
            fallback_message = "Pepani, koma sindingathe kuyankha funso lanu panopa chifukwa chabvuto lina. Kodi pali funso lina lomwe mulinalo?"
        else:
            fallback_message = "I'm sorry, but I can't generate an answer for your query right now. Is there anything else I can assist you with?"
    
        send_whatsapp_message(fallback_message, From)
        return PlainTextResponse(fallback_message)
    except Exception as e:
        print(f"Error handling /fallback endpoint: {e}")
        raise HTTPException(status_code=500, detail="Error processing fallback message")