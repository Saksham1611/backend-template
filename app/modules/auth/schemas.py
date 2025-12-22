from pydantic import BaseModel, EmailStr


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenPayload(BaseModel):
    sub: int | None = None


class LoginRequest(BaseModel):
    username: EmailStr
    password: str
