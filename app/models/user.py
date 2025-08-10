
from pydantic import BaseModel, EmailStr, Field


# Pydantic models

class UserCreate(BaseModel):
    first_name: str = Field(..., min_length=1)
    last_name: str = Field(..., min_length=1)
    email: EmailStr
    password: str = Field(..., min_length=6)

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserInDB(BaseModel):
    first_name: str
    last_name: str
    email: EmailStr
    hashed_password: str




