from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_env: str = "dev"
    port: int = 8000

    # Mongo
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db: str = "tradebot"

    # LLMs / Embeddings
    openai_api_key: str | None = None
    # (optional) if you plan to add Claude later, keep this here:
    anthropic_api_key: str | None = None

    # Zerodha / Kite
    kite_api_key: str | None = None
    kite_api_secret: str | None = None

    # Web search
    tavily_api_key: str | None = None

    # (optional) Twilio if you add WhatsApp/SMS later
    twilio_account_sid: str | None = None
    twilio_auth_token: str | None = None
    twilio_whatsapp_number: str | None = None
    
    # Add this field
    app_url: str = "https://e3030feaeb62.ngrok-free.app"  # Update this with your ngrok/production URL
    # IMPORTANT: ignore extra keys in .env to avoid crashes
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

settings = Settings()
