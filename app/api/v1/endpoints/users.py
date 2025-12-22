from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.dependencies import get_current_superuser, get_current_user, get_db
from app.modules.users.models import User  # Added import for User model
from app.modules.users.schemas import UserCreate, UserRead, UserUpdate
from app.modules.users.service import UserService

router = APIRouter()


@router.post("/", response_model=UserRead, status_code=status.HTTP_201_CREATED)
async def create_user(
    *,
    db: AsyncSession = Depends(get_db),
    user_in: UserCreate,
    current_user: User = Depends(get_current_superuser),
) -> UserRead:
    service = UserService(db)
    user = await service.create(user_in)
    return UserRead.model_validate(user)


@router.get("/{user_id}", response_model=UserRead)
async def read_user_by_id(
    user_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> UserRead:
    service = UserService(db)
    user = await service.get_by_id(user_id)
    return UserRead.model_validate(user)


@router.patch("/{user_id}", response_model=UserRead)
async def update_user(
    *,
    db: AsyncSession = Depends(get_db),
    user_id: int,
    user_in: UserUpdate,
    current_user: User = Depends(get_current_superuser),
) -> UserRead:
    service = UserService(db)
    user = await service.update(user_id, user_in)
    return UserRead.model_validate(user)
