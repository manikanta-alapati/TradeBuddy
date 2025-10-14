# app/services/whatsapp.py
"""
WhatsApp integration using Twilio.
Handles incoming messages and sends responses.
"""

from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
from app.settings import settings
from typing import Optional

import re

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





import re

# app/services/whatsapp.py - REPLACE sanitize_whatsapp_message

def sanitize_whatsapp_message(message: str) -> str:
    """
    Sanitize message for Twilio WhatsApp compatibility.
    
    NUCLEAR OPTION: Remove ALL emojis and formatting.
    This is the only 100% reliable way to avoid error 63038.
    
    Args:
        message: Original message
        
    Returns:
        Plain text message safe for Twilio WhatsApp
    """
    if not message:
        return message
    
    # ============================================
    # STEP 1: Normalize Unicode Punctuation
    # ============================================
    
    punctuation_replacements = {
        ''': "'", ''': "'", '"': '"', '"': '"',
        '–': '-', '—': '-', '―': '-',
        '…': '...',
        '′': "'", '″': '"',
    }
    
    for unicode_char, ascii_char in punctuation_replacements.items():
        message = message.replace(unicode_char, ascii_char)
    
    # ============================================
    # STEP 2: Remove Markdown Formatting
    # ============================================
    
    message = re.sub(r'\*\*(.+?)\*\*', r'\1', message)  # Bold
    message = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'\1', message)  # Italic
    message = re.sub(r'^#{1,6}\s+', '', message, flags=re.MULTILINE)  # Headers
    message = re.sub(r'`(.+?)`', r'\1', message)  # Code
    message = re.sub(r'\[(.+?)\]\((.+?)\)', r'\1', message)  # Links
    
    # ============================================
    # STEP 3: REMOVE ALL EMOJIS (Nuclear Option)
    # ============================================
    
    # Keep ONLY: ASCII + Currency symbols + Basic punctuation
    # Remove: ALL emojis (including "safe" ones)
    
    # This regex keeps only:
    # - Letters (a-z, A-Z)
    # - Numbers (0-9)
    # - Basic punctuation: .,!?;:()[]{}"-'
    # - Whitespace: space, newline, tab
    # - Math operators: +-*/%=<>&|^~
    # - Currency: ₹$€£¥
    # - Special: @#_
    
    allowed_pattern = r'[a-zA-Z0-9\s\-.,!?;:()\[\]{}₹$€£¥\n\r\t+\-*/%=<>&|^~"\'`@#_]'
    
    # Filter message character by character
    cleaned_chars = []
    for char in message:
        if re.match(allowed_pattern, char):
            cleaned_chars.append(char)
        # Silently skip everything else (including ALL emojis)
    
    message = ''.join(cleaned_chars)
    
    # ============================================
    # STEP 4: Clean Up Formatting
    # ============================================
    
    # Remove multiple spaces
    message = re.sub(r' {2,}', ' ', message)
    
    # Remove excessive newlines
    message = re.sub(r'\n{3,}', '\n\n', message)
    
    # Remove spaces at start/end of lines
    message = '\n'.join(line.strip() for line in message.split('\n'))
    
    return message.strip()