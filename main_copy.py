# Third-party imports
from langdetect import detect
from fastapi import FastAPI, Form, Depends, Request
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
from fastapi.responses import PlainTextResponse
from decouple import config

# Find your Account SID and Auth Token at twilio.com/console
# and set the environment variables. See http://twil.io/secure
ACCOUNT_SID = config("TWILIO_ACCOUNT_SID")
AUTH_TOKEN = config("TWILIO_AUTH_TOKEN")
OPENAI_API_KEY = config("OPENAI_API_KEY")
CLIENT = Client(ACCOUNT_SID, AUTH_TOKEN,timeout=30)
TWILIO_NUMBER = config('TWILIO_NUMBER')

# Internal imports
from models import Conversation, SessionLocal
from utils import logger, run_rag_query, detect_language
from sql_chain import run_sql_chain
import logging

app = FastAPI()

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

def handle_incoming_message(incoming_msg, sender):
    # Implement your chatbot logic here
    response_text = "Hello, thanks for your message. We will get back to you shortly."

    # Send a response using the Twilio Client
    send_whatsapp_message(response_text, sender)

    return response_text

@app.post("/sms")
async def sms_reply(request: Request):
    form = await request.form()
    incoming_msg = form.get('Body')
    sender = form.get('From')

    # Handle the incoming message and send a response
    response_text = handle_incoming_message(incoming_msg, sender)

    # Create a Twilio response object
    resp = MessagingResponse()
    resp.message(response_text)

    return str(resp)

# Dependency
def get_db():
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()

@app.post("/message")
async def reply(request: Request, Body: str = Form(), db: Session = Depends(get_db)):
    logger.info('Senging WhatsApp Mesage')
    # Extract the phone number from the incoming webhook request
    form_data = await request.form()
    whatsapp_number = form_data['From'].split("whatsapp:")[-1]
    print(f"Sending the LangChain response to this number: {whatsapp_number}")

    # Get the generated text from the LangChain agent

    # uses narrative from report (text files to provide answers)
    # try:
    #     text_response = run_rag_query(Body)
    #     print("Response from RAG:", text_response)
    # except Exception as e:
    #     print(e)
    
    text_response = "No text response"
    try:
        sql_response = run_sql_chain(Body)
        print("Response from RAG:", text_response)
    except Exception as e:
        print(e)
    

    if "I don't know" in text_response:
        combined_response = sql_response
    else:
        combined_response = text_response + "\n" + sql_response

    print(combined_response)
    # try:
    #     if "I don't know" in text_response:
    #         # uses tables in database to provide answrs based on tabular data
    #         try:
    #             langchain_response = run_sql_chain(Body)
    #             print(langchain_response)
    #         except Exception as e:
    #             print(e)
    #     else:
    #         langchain_response = text_response
    # except Exception as e:
    #     print(e)
    #     reply_fallback()
    #     quit()
    #     # langchain_response = "I'm sorry, something went wrong, please try again later"

    # Store the conversation in the database
    try:
        conversation = Conversation(
            sender=whatsapp_number,
            message=Body,
            response=combined_response
            )
        db.add(conversation)
        db.commit()
        logger.info(f"Conversation #{conversation.id} stored in database")
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Error storing conversation in database: {e}")
    
    # Now send the message
    try:
        response = MessagingResponse()
        msg = response.message()
        msg.body(combined_response)
        xml_response = str(response)
        logging.info(f"Outgoing response: {xml_response}")
        return PlainTextResponse(xml_response, media_type="application/xml")
    except Exception as e:
        logger.error(f"Error sending message to {whatsapp_number}: {e}")
        return PlainTextResponse("Error processing request", status_code=500)


@app.post("/fallback")
async def reply_fallback(request: Request, Body: str = Form(), db: Session = Depends(get_db)):
    logger.info('Sending WhatsApp Mesage')
    # Extract the phone number from the incoming webhook request
    form_data = await request.form()
    whatsapp_number = form_data['From'].split("whatsapp:")[-1]
    print(f"Sending the LangChain response to this number: {whatsapp_number}")

    # Message to send when there is a problem
    detected_lan = detect_language(Body)
    if detected_lan == "sw" or detected_lan  == "ny":
        fallback_message = "Pepani, koma sindingathe kuyankha funso lanu panopa chifukwa chabvuto lina. Kodi pali funso lina lomwe mulinalo?"
    else:
        fallback_message = "I'm sorry, but I can't generate an answer for your query right now. Is there anything else I can assist you with?"
        
    # Now send the message
    try:
        response = MessagingResponse()
        msg = response.message()
        msg.body(fallback_message )
        xml_response = str(response)
        logging.info(f"Outgoing response: {xml_response}")
        return PlainTextResponse(xml_response, media_type="application/xml")
    except Exception as e:
        logger.error(f"Error sending message to {whatsapp_number}: {e}")
        return PlainTextResponse("Error processing request", status_code=500)

