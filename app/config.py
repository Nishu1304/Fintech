from pydantic_settings import BaseSettings  # use 'from pydantic import BaseSettings' for Pydantic v1
from fastapi.middleware.cors import CORSMiddleware

class Settings(BaseSettings):
    mongo_uri: str = "mongodb://localhost:27017"  # default value for development
    mongo_db_name: str = "users"
    # add more project/global settings here

    class Config:
        env_file = ".env"  # load values from .env file

settings = Settings()

# Simple in-memory session store
SESSION_STORE = {}

def add_cors_middleware(app):
    origins = ["*"] # Allowing all origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )