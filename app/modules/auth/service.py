from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import UnauthorizedError
from app.core.security import create_access_token, verify_password
from app.modules.auth.schemas import Token
from app.modules.users.service import UserService


class AuthService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.user_service = UserService(db)

    async def authenticate(self, email: str, password: str) -> Token:
        user = await self.user_service.get_by_email(email)
        if not user or not verify_password(password, user.hashed_password):
            raise UnauthorizedError("Incorrect email or password")

        if not user.is_active:
            raise UnauthorizedError("Inactive user")

        access_token = create_access_token(data={"sub": str(user.id)})
        return Token(access_token=access_token, token_type="bearer")
