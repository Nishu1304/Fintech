# Helper functions
from typing import Optional
from passlib.context import CryptContext
from motor.motor_asyncio import AsyncIOMotorClient
from app.models.user import UserCreate, UserInDB
from app.config import settings



client = AsyncIOMotorClient(settings.mongo_uri)
db = client[settings.mongo_db_name]
users_collection = db.get_collection("users")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

async def get_user_by_email(email: str) -> Optional[UserInDB]:
    user = await users_collection.find_one({"email": email})
    if user:
        return UserInDB(**user)

async def create_user(user_data: UserCreate) -> UserInDB:
    user_dict = user_data.model_dump()
    user_dict["hashed_password"] = hash_password(user_dict.pop("password"))
    await users_collection.insert_one(user_dict)
    return UserInDB(**user_dict)