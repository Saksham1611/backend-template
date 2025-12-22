from pydantic import BaseModel, ConfigDict, EmailStr


# Shared properties
class UserBase(BaseModel):
    email: EmailStr | None = None
    is_active: bool | None = True
    is_superuser: bool = False


# Properties to receive via API on creation
class UserCreate(UserBase):
    email: EmailStr
    password: str


# Properties to receive via API on update
class UserUpdate(UserBase):
    password: str | None = None


class UserRead(UserBase):
    id: int

    model_config = ConfigDict(from_attributes=True)
