from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ConflictError, NotFoundError
from app.core.security import get_password_hash
from app.modules.users.models import User
from app.modules.users.schemas import UserCreate, UserUpdate


class UserService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_by_id(self, user_id: int) -> User:
        user = await self.db.get(User, user_id)
        if not user:
            raise NotFoundError("User", user_id)
        return user

    async def get_by_email(self, email: str) -> User | None:
        result = await self.db.execute(select(User).where(User.email == email))
        return result.scalars().first()

    async def create(self, data: UserCreate) -> User:
        if await self.get_by_email(data.email):
            raise ConflictError(f"User with email {data.email} already exists")

        user = User(
            email=data.email,
            hashed_password=get_password_hash(data.password),
            is_active=data.is_active,
            is_superuser=data.is_superuser,
        )
        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)
        return user

    async def update(self, user_id: int, data: UserUpdate) -> User:
        user = await self.get_by_id(user_id)
        update_data = data.model_dump(exclude_unset=True)

        if password := update_data.pop("password", None):
            update_data["hashed_password"] = get_password_hash(password)

        for field, value in update_data.items():
            setattr(user, field, value)

        await self.db.commit()
        await self.db.refresh(user)
        return user
