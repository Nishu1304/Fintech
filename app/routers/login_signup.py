from fastapi import APIRouter, HTTPException, status
from app.models.user import UserCreate, UserLogin
from app.crud.user import get_user_by_email, create_user, verify_password

app = APIRouter()


@app.post("/signup", status_code=status.HTTP_201_CREATED)
async def signup(user: UserCreate):
    existing_user = await get_user_by_email(user.email)
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    created_user = await create_user(user)
    return {"msg": "User created successfully", "email": created_user.email}

@app.post("/login")
async def login(user: UserLogin):
    db_user = await get_user_by_email(user.email)
    if not db_user or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return {"msg": f"Welcome {db_user.first_name}!"}