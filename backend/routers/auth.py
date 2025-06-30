from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from passlib.context import CryptContext
import asyncpg

router = APIRouter()

DATABASE_URL = "данные для подключение к Вашей БД"
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class UserRegister(BaseModel):
    username: str
    password: str

async def connect_db():
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        return conn
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.post("/login-or-register")
async def login_or_register(user: UserRegister):
    conn = await connect_db()
    try:
        user_in_db = await conn.fetchrow("SELECT * FROM users WHERE username = $1", user.username)
        if not user_in_db:
            hashed_password = pwd_context.hash(user.password)
            await conn.execute(
                "INSERT INTO users (username, password, first_name, last_name) VALUES ($1, $2, '', '')",
                user.username, hashed_password
            )
            return {"message": "User created successfully"}
        return {"message": "User already exists, no changes made"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
    finally:
        await conn.close()