# app/services/whatsapp.py
"""
WhatsApp integration using Twilio.
Handles incoming messages and sends responses.
"""

from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
from app.settings import settings
from typing import Optional

# Initialize Twilio client
_twilio_client: Optional[Client] = None

def get_twilio_client() -> Client:
    """Get or create Twilio client."""
    global _twilio_client
    if _twilio_client is None:
        if not settings.twilio_account_sid or not settings.twilio_auth_token:
            raise RuntimeError("Twilio credentials missing in .env")
        _twilio_client = Client(
            settings.twilio_account_sid,
            settings.twilio_auth_token
        )
    return _twilio_client


def send_whatsapp_message(to_number: str, message: str) -> dict:
    """
    Send a WhatsApp message via Twilio.
    
    Args:
        to_number: Recipient WhatsApp number (format: whatsapp:+1234567890)
        message: Message text to send
    
    Returns:
        Dict with message SID and status
    """
    client = get_twilio_client()
    
    # Ensure number has whatsapp: prefix
    if not to_number.startswith("whatsapp:"):
        to_number = f"whatsapp:{to_number}"
    
    try:
        msg = client.messages.create(
            from_=settings.twilio_whatsapp_number,
            to=to_number,
            body=message
        )
        return {
            "success": True,
            "sid": msg.sid,
            "status": msg.status
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def parse_incoming_whatsapp(form_data: dict) -> dict:
    """
    Parse incoming WhatsApp webhook data from Twilio.
    
    Args:
        form_data: Form data from Twilio webhook
    
    Returns:
        Dict with parsed message info
    """
    return {
        "from": form_data.get("From", "").replace("whatsapp:", ""),
        "to": form_data.get("To", "").replace("whatsapp:", ""),
        "body": form_data.get("Body", "").strip(),
        "message_sid": form_data.get("MessageSid"),
        "num_media": int(form_data.get("NumMedia", 0)),
        "profile_name": form_data.get("ProfileName", "User")
    }


def create_twiml_response(message: str) -> str:
    """
    Create TwiML response for Twilio webhook.
    
    Args:
        message: Response message text
    
    Returns:
        TwiML XML string
    """
    response = MessagingResponse()
    response.message(message)
    return str(response)