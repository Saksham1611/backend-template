# Backend Development Gospel

The definitive guide for building clean, scalable FastAPI backends with proper database patterns, modular architecture, and production-ready configurations.

---

## Folder Structure

```
wellnest-template-service/
├── pyproject.toml              # Modern dependency management (uv)
├── .pre-commit-config.yaml
├── .env.example
├── alembic.ini                 # DB Migrations config
├── app/
│   ├── main.py                 # Entry point + Lifespan events
│   ├── core/
│   │   ├── config.py           # Pydantic Settings
│   │   ├── exceptions.py       # Custom HTTP exceptions
│   │   ├── logging.py          # Rich logging setup
│   │   └── security.py         # JWT + password hashing
│   ├── db/
│   │   ├── session.py          # Engine & SessionLocal
│   │   └── base.py             # The "Registry" file (imports all models)
│   ├── api/
│   │   └── v1/
│   │       ├── router.py       # The "Switchboard" - aggregates endpoints
│   │       ├── dependencies.py # The "Bouncers" (get_db, get_current_user)
│   │       └── endpoints/      # API route handlers
│   │           ├── auth.py     # Uses modules.auth.service
│   │           └── bookings.py # Uses modules.bookings.service
│   └── modules/                # PURE BUSINESS LOGIC (No API code!)
│       ├── auth/
│       │   ├── models.py       # User table
│       │   ├── schemas.py      # Pydantic request/response
│       │   └── service.py      # Business logic
│       └── bookings/
│           ├── models.py       # Booking table
│           ├── schemas.py
│           └── service.py
├── tests/                      # ALL TESTS LIVE HERE
│   ├── __init__.py
│   ├── conftest.py             # ⚡ Shared fixtures (DB, client, mocks)
│   ├── unit/                   # Fast tests - pure logic, no DB
│   │   └── test_modules/
│   │       ├── test_auth_service.py
│   │       └── test_booking_service.py
│   └── integration/            # Slow tests - API endpoints, uses Test DB
│       └── api/
│           └── v1/
│               └── test_auth_endpoints.py
└── migrations/                 # Alembic folder
    └── env.py                  # Imports app.db.base
```

> [!IMPORTANT]
> **Key Change:** Routers live in `api/v1/endpoints/`, NOT in `modules/`.
> Modules contain ONLY business logic (models, schemas, services).

---

## Component Deep Dives

Each component in this structure has a specific purpose. Understanding **what**, **when**, and **how** to use each one is critical for maintaining clean architecture.

---

### 📁 `main.py` - The Entry Point

**What it is:** The single entry point that creates the FastAPI application instance and wires everything together.

**When to touch it:**
- Adding new middleware (CORS, logging, error handlers)
- Mounting routers from `api/v1/`
- Adding startup/shutdown events (e.g., DB pool warmup, cache connections)

**What NOT to put here:**
- Business logic
- Route handlers
- Database queries

```python
# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.core.config import settings
from app.core.logging import setup_logging
from app.api.v1.router import api_router

# Setup Rich logging BEFORE anything else
setup_logging(level="INFO")
logger = logging.getLogger("app")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[bold green]🚀 Starting up...[/]", extra={"markup": True})
    yield
    logger.info("[bold red]🛑 Shutting down...[/]", extra={"markup": True})

app = FastAPI(
    title=settings.APP_NAME,
    openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
    lifespan=lifespan,
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the API router (the "switchboard")
app.include_router(api_router, prefix=settings.API_V1_PREFIX)

@app.get("/health")
def health_check():
    return {"status": "healthy"}
```

---

### 📁 `core/` - Application Core

The `core/` folder contains **cross-cutting concerns** that don't belong to any single domain.

#### `core/config.py` - Settings Management

**What it is:** Centralized configuration using `pydantic-settings`.

**When to add here:**
- New environment variables
- API keys for external services
- Feature flags

**Pattern:** All settings are validated at startup. Missing required values = app won't start.

---

#### `core/exceptions.py` - Custom Exceptions

**What it is:** Application-specific exceptions that translate to HTTP responses.

**When to add here:**
- New error types (e.g., `ResourceNotFoundError`, `PermissionDeniedError`)
- Custom error responses with specific status codes

```python
# app/core/exceptions.py
from fastapi import HTTPException, status

class NotFoundError(HTTPException):
    def __init__(self, resource: str, resource_id: int | str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{resource} with id '{resource_id}' not found",
        )

class UnauthorizedError(HTTPException):
    def __init__(self, message: str = "Not authenticated"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=message,
            headers={"WWW-Authenticate": "Bearer"},
        )

class ForbiddenError(HTTPException):
    def __init__(self, message: str = "Permission denied"):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=message,
        )

class ConflictError(HTTPException):
    def __init__(self, message: str):
        super().__init__(
            status_code=status.HTTP_409_CONFLICT,
            detail=message,
        )
```

---

#### `core/security.py` - Authentication & Authorization

**What it is:** JWT creation, password hashing, token verification.

**When to add here:**
- OAuth2 flows
- Password hashing utilities
- Token generation/verification

```python
# app/core/security.py
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from app.core.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

def decode_access_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
    except JWTError:
        return None
```

---

#### `core/logging.py` - Rich Logging

**What it is:** Beautiful, colorful console logging using the `rich` library.

**Why it matters:** FastAPI uses standard Python `logging` by default (uvicorn access logs). Rich intercepts those to make everything look pretty, not just your custom logs.

```python
# app/core/logging.py
import logging
import sys
from rich.logging import RichHandler

def setup_logging(level: str = "INFO"):
    """
    Replaces standard logging handlers with Rich.
    Call this ONCE at app startup, before anything else.
    """
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Remove existing handlers (like the default Uvicorn one)
    # so we don't get double logs
    logger.handlers = []

    # Create the Rich Handler
    rich_handler = RichHandler(
        rich_tracebacks=True,  # Pretty colorful tracebacks
        markup=True,           # Allow [bold red]text[/] syntax
        show_time=True,
        show_path=False,
    )

    # Format
    formatter = logging.Formatter("%(message)s")
    rich_handler.setFormatter(formatter)

    logger.addHandler(rich_handler)

    # Silence noisy libraries if needed
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
```

**Usage in your code:**

```python
import logging
logger = logging.getLogger("app.bookings")

logger.info("Booking created for user %s", user_id)
logger.warning("[bold yellow]Low availability![/]", extra={"markup": True})
logger.error("[bold red]Payment failed[/]", extra={"markup": True})
```

> [!TIP]
> Use `extra={"markup": True}` to enable Rich markup syntax like `[bold green]text[/]`.

---

### 📁 `db/` - Database Layer

#### `db/session.py` - Engine & Session Factory

**What it is:** Creates the SQLAlchemy engine, session factory, and `Base` class.

**When to touch it:**
- Changing database connection settings
- Adding connection pooling options
- Modifying `get_db()` dependency behavior

> [!WARNING]
> Never create multiple `Base` classes. Always import from `db/session.py`.

---

#### `db/base.py` - Model Registry

**What it is:** A single file that imports ALL models so Alembic can detect them.

**When to touch it:**
- Every time you create a new model in any module

**The Pattern:**
```python
# app/db/base.py
from app.db.session import Base  # noqa

# Import ALL models here
from app.modules.auth.models import User  # noqa
from app.modules.bookings.models import Booking  # noqa
from app.modules.payments.models import Payment  # noqa
```

---

### 📁 `api/v1/` - The API Layer

This is where HTTP meets your business logic. Think of it like a **power strip** analogy.

#### The Wiring Concept

```mermaid
flowchart LR
    A[main.py] -->|"plugs in"| B[api/v1/router.py]
    B -->|"plugs in"| C[endpoints/auth.py]
    B -->|"plugs in"| D[endpoints/bookings.py]
    C -->|"uses"| E[modules/auth/service.py]
    D -->|"uses"| F[modules/bookings/service.py]
```

---

#### Step 1: The Endpoint (The "Plug")

Each file in `endpoints/` defines routes for one domain. It imports from `modules/` for business logic.

```python
# app/api/v1/endpoints/auth.py
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.v1.dependencies import get_db
from app.modules.auth.service import AuthService
from app.modules.auth.schemas import LoginRequest, TokenResponse

router = APIRouter()  # This is the "plug"

@router.post("/login", response_model=TokenResponse)
def login(data: LoginRequest, db: Session = Depends(get_db)):
    service = AuthService(db)
    return service.authenticate(data.email, data.password)

@router.post("/register")
def register(data: RegisterRequest, db: Session = Depends(get_db)):
    service = AuthService(db)
    return service.create_user(data)
```

---

#### Step 2: The Switchboard (`router.py`)

This file collects all the plugs from `endpoints/` and wires them together.

```python
# app/api/v1/router.py
from fastapi import APIRouter
from app.api.v1.endpoints import auth, bookings

api_router = APIRouter()

# Plug them in
api_router.include_router(auth.router, prefix="/auth", tags=["Auth"])
api_router.include_router(bookings.router, prefix="/bookings", tags=["Bookings"])
```

---

#### Step 3: The Main Power (`main.py`)

```python
# Plug the switchboard into the wall
app.include_router(api_router, prefix="/api/v1")
```

> [!NOTE]
> **Why this structure?** It keeps `modules/` pure (no FastAPI code), making them reusable for CLI tools, background jobs, or other interfaces.

---

#### `api/v1/dependencies.py` - The "Bouncers"

**ELI5:** Dependencies are "Bouncers" and "Assistants". Before a user enters your API route function, they have to get past the dependencies.

**Common things found here:**

| Dependency | Role |
|------------|------|
| `get_db` | 🚪 **Database Assistant** - Opens a connection, hands it to you, closes it when done |
| `get_current_user` | 🔐 **Security Bouncer** - "Show me your JWT. Okay, you're User #42." |
| `get_pagination` | 📄 **Pagination Assistant** - Sets defaults for page/limit |

**Why separate it?** If you put `get_current_user` inside `endpoints/auth.py`, you can't easily use it in `endpoints/bookings.py` without circular imports.

```python
# app/api/v1/dependencies.py
from typing import Generator
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.core.security import decode_access_token
from app.modules.auth.models import User
from app.core.exceptions import UnauthorizedError

# 1. The Database Assistant
def get_db() -> Generator:
    """Opens DB connection, yields it, closes when done."""
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()

# 2. The Security Bouncer
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """Decodes JWT, fetches user, or raises 401."""
    payload = decode_access_token(token)
    if payload is None:
        raise UnauthorizedError("Invalid token")

    user_id = payload.get("sub")
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise UnauthorizedError("User not found")

    return user

def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """Extra check: is the user active?"""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# 3. The Pagination Assistant
def get_pagination(skip: int = 0, limit: int = 20) -> dict:
    """Default pagination params."""
    return {"skip": skip, "limit": min(limit, 100)}  # Cap at 100
```

---

### 📁 `modules/` - Pure Business Logic

This is where **business logic lives**. Each module represents a bounded context (domain).

> [!IMPORTANT]
> **NO FastAPI code here!** No routers, no `Depends()`, no HTTP concerns.
> This keeps modules reusable for CLI tools, background jobs, tests, etc.

#### Module Structure

```
modules/
└── bookings/
    ├── models.py     # SQLAlchemy table definition
    ├── schemas.py    # Pydantic request/response schemas
    └── service.py    # Business logic (CRUD + workflows)
```

---

#### `models.py` - Database Tables

**What it is:** SQLAlchemy ORM model for this domain's database table(s).

**Rules:**
- One model per table (can have multiple tables per module)
- Use string references for cross-module relationships
- No business logic here

---

#### `schemas.py` - Request/Response Shapes

**What it is:** Pydantic models for API input/output validation.

**Pattern:** Separate schemas for different operations.

```python
# app/modules/bookings/schemas.py
from pydantic import BaseModel
from datetime import datetime

# Base schema with shared fields
class BookingBase(BaseModel):
    therapist_id: int
    start_time: datetime
    duration_minutes: int = 60

# Schema for creating (input)
class BookingCreate(BookingBase):
    pass

# Schema for updating (input) - all fields optional
class BookingUpdate(BaseModel):
    start_time: datetime | None = None
    duration_minutes: int | None = None

# Schema for reading (output) - includes DB fields
class BookingRead(BookingBase):
    id: int
    user_id: int
    status: str
    created_at: datetime

    class Config:
        from_attributes = True  # Enables ORM mode
```

---

#### `service.py` - Business Logic

**What it is:** The brains of the module. All CRUD operations, workflows, and cross-model queries.

**Rules:**
- Services receive a `db: Session` in constructor
- Services can import models from other modules
- Keep routes thin - move logic to services

```python
# app/modules/bookings/service.py
from sqlalchemy.orm import Session
from app.modules.bookings.models import Booking
from app.modules.bookings.schemas import BookingCreate, BookingUpdate
from app.core.exceptions import NotFoundError, ConflictError

class BookingService:
    def __init__(self, db: Session):
        self.db = db

    def get_by_id(self, booking_id: int) -> Booking:
        booking = self.db.query(Booking).filter(Booking.id == booking_id).first()
        if not booking:
            raise NotFoundError("Booking", booking_id)
        return booking

    def get_user_bookings(self, user_id: int) -> list[Booking]:
        return self.db.query(Booking).filter(Booking.user_id == user_id).all()

    def create(self, user_id: int, data: BookingCreate) -> Booking:
        # Check for conflicts
        existing = self.db.query(Booking).filter(
            Booking.therapist_id == data.therapist_id,
            Booking.start_time == data.start_time,
        ).first()
        if existing:
            raise ConflictError("This time slot is already booked")

        booking = Booking(user_id=user_id, **data.model_dump())
        self.db.add(booking)
        self.db.commit()
        self.db.refresh(booking)
        return booking

    def update(self, booking_id: int, data: BookingUpdate) -> Booking:
        booking = self.get_by_id(booking_id)
        update_data = data.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(booking, key, value)
        self.db.commit()
        self.db.refresh(booking)
        return booking

    def delete(self, booking_id: int) -> None:
        booking = self.get_by_id(booking_id)
        self.db.delete(booking)
        self.db.commit()
```

---

#### `router.py` - API Endpoints

**What it is:** FastAPI route handlers that wire HTTP to services.

**Rules:**
- Keep handlers thin - they should just call services
- Use dependencies for auth/db
- Return schemas, not models directly

```python
# app/modules/bookings/router.py
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.api.v1.dependencies import get_current_user
from app.modules.auth.models import User
from app.modules.bookings.service import BookingService
from app.modules.bookings.schemas import BookingCreate, BookingRead, BookingUpdate

router = APIRouter()

@router.get("/", response_model=list[BookingRead])
def get_my_bookings(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    service = BookingService(db)
    return service.get_user_bookings(current_user.id)

@router.post("/", response_model=BookingRead, status_code=status.HTTP_201_CREATED)
def create_booking(
    data: BookingCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    service = BookingService(db)
    return service.create(current_user.id, data)

@router.get("/{booking_id}", response_model=BookingRead)
def get_booking(
    booking_id: int,
    db: Session = Depends(get_db),
):
    service = BookingService(db)
    return service.get_by_id(booking_id)

@router.patch("/{booking_id}", response_model=BookingRead)
def update_booking(
    booking_id: int,
    data: BookingUpdate,
    db: Session = Depends(get_db),
):
    service = BookingService(db)
    return service.update(booking_id, data)

@router.delete("/{booking_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_booking(
    booking_id: int,
    db: Session = Depends(get_db),
):
    service = BookingService(db)
    service.delete(booking_id)
```

---

### 📁 `migrations/` - Database Migrations

**What it is:** Alembic migration files that version your database schema.

**Key Files:**
- `alembic.ini` - Configuration (DB URL, script location)
- `migrations/env.py` - Must import `app.db.base` to detect all models
- `migrations/versions/` - Auto-generated migration scripts

**Common Commands:**
```bash
# Generate a new migration
alembic revision --autogenerate -m "Add payments table"

# Apply all pending migrations
alembic upgrade head

# Rollback one migration
alembic downgrade -1

# See current migration
alembic current
```

---

### 📁 `tests/` - Testing with Pytest

Don't mix tests inside your `app/` folder. Keep them in a root `tests/` directory.

> [!TIP]
> Tests are split into **unit** (fast, no DB) and **integration** (slower, uses Test DB).

---

#### `conftest.py` - The Setup Wizard

This is the most important file. It sets up your Test Environment with **Dependency Overrides**.

We need three main things:
1. **Test Database** - A fresh, empty database for every test session
2. **Async Client** - The HTTP client to hit your API
3. **Fixtures** - Common objects like a "fake user" or "admin token"

```python
# tests/conftest.py
import pytest
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from app.main import app
from app.db.base import Base
from app.api.v1.dependencies import get_db, get_current_user

# 1. SETUP TEST DATABASE
# Use SQLite for speed, or a separate Postgres DB for accuracy
TEST_DATABASE_URL = "sqlite+aiosqlite:///./test.db"

engine = create_async_engine(TEST_DATABASE_URL, echo=False)
TestingSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

@pytest.fixture(scope="session", autouse=True)
async def setup_test_db():
    """Create tables once before tests start, drop them after."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

# 2. DATABASE SESSION FIXTURE
# This overrides the real `get_db` dependency
async def override_get_db():
    async with TestingSessionLocal() as session:
        yield session

app.dependency_overrides[get_db] = override_get_db

# 3. ASYNC CLIENT FIXTURE
@pytest.fixture
async def client():
    """Giving you a client that talks to your FastAPI app"""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as c:
        yield c

# 4. MOCK USER FIXTURE (Optional but useful)
@pytest.fixture
def mock_user_dependency():
    """If a test needs a logged-in user, use this override"""
    fake_user = {"id": 1, "email": "test@wellnest.com", "role": "admin"}
    app.dependency_overrides[get_current_user] = lambda: fake_user
    yield
    # Clean up after test
    app.dependency_overrides.pop(get_current_user, None)
```

---

#### Unit Tests - Testing `modules/`

Test **service logic directly**. No HTTP, no routers, no URLs. Just "input → output".

```python
# tests/unit/test_modules/test_booking_service.py
import pytest
from app.modules.bookings.service import calculate_total_price

# Simple logic test - No Database needed!
def test_calculate_total_price():
    price = 100
    tax_rate = 0.1

    total = calculate_total_price(price, tax_rate)

    assert total == 110.0

# Service test with Mocked DB
@pytest.mark.asyncio
async def test_create_booking_logic(mocker):
    # Mock the DB session so we don't hit the real DB
    mock_db = mocker.MagicMock()

    from app.modules.bookings.service import BookingService
    service = BookingService(mock_db)
    await service.create_booking(user_id=1, date="2025-12-25")

    # Assert that the DB 'add' method was called
    mock_db.add.assert_called_once()
```

---

#### Integration Tests - Testing `api/`

Test **endpoints**. Use the `client` fixture from conftest. This ensures Router + Pydantic + Dependencies all work together.

```python
# tests/integration/api/v1/test_auth_endpoints.py
import pytest
from httpx import AsyncClient

# This test uses the real Test DB (via conftest override)
@pytest.mark.asyncio
async def test_register_user(client: AsyncClient):
    payload = {
        "email": "newuser@example.com",
        "password": "strongpassword123",
        "full_name": "Test User"
    }

    response = await client.post("/api/v1/auth/register", json=payload)

    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "newuser@example.com"
    assert "id" in data

# This test forces a Mock Login state
@pytest.mark.asyncio
async def test_get_me_endpoint(client: AsyncClient, mock_user_dependency):
    # Because we used the fixture 'mock_user_dependency',
    # FastAPI thinks we are already logged in!

    response = await client.get("/api/v1/auth/me")

    assert response.status_code == 200
    assert response.json()["email"] == "test@wellnest.com"
```

---

#### Test Commands

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/

# Run a specific test file
pytest tests/unit/test_modules/test_booking_service.py

# Run tests matching a pattern
pytest -k "test_register"

# Show print statements
pytest -s

# Run with coverage report
pytest --cov=app --cov-report=html
```

---

#### Test Dependencies (pyproject.toml)

```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "pytest-mock>=3.10.0",
    "pytest-cov>=4.0.0",
    "httpx>=0.26.0",
    "aiosqlite>=0.19.0",  # For async SQLite testing
]
```

Install with: `uv add --dev pytest pytest-asyncio httpx pytest-mock pytest-cov aiosqlite`

---

## Interlinked DB Logic - The 3 Rules

When splitting database models across modules, follow these 3 golden rules to avoid circular imports and maintain clean architecture.

### Rule 1: Define Models Locally, Register Centrally

**Don't** put all tables in one file. Put domain-specific tables in their respective module's `models.py`.

**However**, to ensure Alembic sees them all, create a central registry in `app/db/base.py`:

```python
# app/db/base.py
from app.db.session import Base  # The SQLAlchemy Declarative Base

# Import all your models here so Alembic finds them
from app.modules.auth.models import User  # noqa
from app.modules.bookings.models import Booking  # noqa
# Add new models here as you create them
```

> [!IMPORTANT]
> Every new model must be imported in `base.py`, otherwise Alembic won't detect it during migrations.

---

### Rule 2: Use "String" Relationships (No Cross-Model Imports)

**Never** import one model into another model file. This causes circular imports.

Instead, use **string references** for SQLAlchemy relationships:

```python
# app/modules/bookings/models.py
from app.db.session import Base
from sqlalchemy import Column, Integer, ForeignKey
from sqlalchemy.orm import relationship

class Booking(Base):
    __tablename__ = "bookings"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))  # Refers to TABLE name

    # Use string "User", not the class User
    user = relationship("User", back_populates="bookings")
```

```python
# app/modules/auth/models.py
from app.db.session import Base
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True)

    # Use string "Booking", not the class Booking
    bookings = relationship("Booking", back_populates="user")
```

> [!TIP]
> SQLAlchemy resolves string references at runtime, so circular imports are avoided entirely.

---

### Rule 3: Cross-Service Logic Belongs in Services, Not Models

If you need to "Get a user and check their bookings", **don't** do it in the Model. Do it in the Service.

```python
# app/modules/bookings/service.py
from sqlalchemy.orm import Session
from app.modules.bookings.models import Booking
from app.modules.auth.models import User  # Safe to import in service layer

class BookingService:
    def __init__(self, db: Session):
        self.db = db

    def create_booking(self, user_id: int, **booking_data):
        # Even though code is split, the DB session is shared
        # You can query different tables freely
        user = self.db.query(User).get(user_id)
        if not user:
            raise ValueError("User not found")

        booking = Booking(user_id=user.id, **booking_data)
        self.db.add(booking)
        self.db.commit()
        self.db.refresh(booking)
        return booking
```

> [!NOTE]
> Importing models in **services** is perfectly safe because services are not imported by models.

---

## Core Configuration (pydantic-settings)

Modern configuration management using `pydantic-settings` for type-safe, validated environment variables.

```python
# app/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import Optional

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App
    APP_NAME: str = "WellNest Service"
    DEBUG: bool = False
    API_V1_PREFIX: str = "/api/v1"

    # Database
    DATABASE_URL: str
    DB_ECHO: bool = False  # SQL query logging

    # Auth / Security
    SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    ALGORITHM: str = "HS256"

    # External Services (optional)
    REDIS_URL: Optional[str] = None
    SENTRY_DSN: Optional[str] = None

    # CORS
    CORS_ORIGINS: list[str] = ["http://localhost:3000"]


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance - loaded once at startup."""
    return Settings()


settings = get_settings()
```

**Usage:**
```python
from app.core.config import settings

print(settings.DATABASE_URL)
print(settings.APP_NAME)
```

---

## Database Session Setup

```python
# app/db/session.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.core.config import settings

engine = create_engine(
    settings.DATABASE_URL,
    echo=settings.DB_ECHO,
    pool_pre_ping=True,  # Reconnect on stale connections
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """Dependency for FastAPI routes."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

---

## Alembic Integration

For Alembic to detect all models, update `migrations/env.py`:

```python
# migrations/env.py
from app.db.base import Base  # This imports all models via base.py
from app.core.config import settings

target_metadata = Base.metadata
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)
```

---

## Quick Reference Diagram

```mermaid
graph TD
    A[app/db/base.py] -->|imports| B[auth/models.py]
    A -->|imports| C[bookings/models.py]

    B -->|string ref| C
    C -->|string ref| B

    D[bookings/service.py] -->|imports| B
    D -->|imports| C

    E[migrations/env.py] -->|imports| A
```

---

## File Responsibility Summary

| File | Purpose |
|------|---------|
| `core/config.py` | Environment variables, settings validation |
| `core/logging.py` | Rich logging setup |
| `core/exceptions.py` | Custom HTTP exception classes |
| `core/security.py` | JWT creation, password hashing |
| `db/session.py` | Engine, SessionLocal, Base |
| `db/base.py` | Central model registry for Alembic |
| `modules/*/models.py` | Domain-specific SQLAlchemy models |
| `modules/*/schemas.py` | Pydantic request/response schemas |
| `modules/*/service.py` | Business logic, cross-model queries |
| `api/v1/endpoints/*.py` | FastAPI route handlers |
| `api/v1/router.py` | The "switchboard" - aggregates endpoints |
| `api/v1/dependencies.py` | The "bouncers" (`get_db`, `get_current_user`) |
| `tests/conftest.py` | Test fixtures, DB override, mock user |
| `tests/unit/` | Fast tests for pure business logic |
| `tests/integration/` | Endpoint tests using Test DB |

---

## Patterns vs Anti-Patterns

### ✅ DO: Keep Routers Thin

```python
# GOOD - Router just calls service
@router.post("/")
def create_booking(data: BookingCreate, db: Session = Depends(get_db)):
    service = BookingService(db)
    return service.create(data)
```

### ❌ DON'T: Put Logic in Routers

```python
# BAD - Business logic in router
@router.post("/")
def create_booking(data: BookingCreate, db: Session = Depends(get_db)):
    existing = db.query(Booking).filter(...).first()  # ❌ Query in router
    if existing:
        raise HTTPException(409, "Conflict")  # ❌ Logic in router
    booking = Booking(**data.dict())
    db.add(booking)
    db.commit()  # ❌ DB operations in router
    return booking
```

---

### ✅ DO: Use String References in Models

```python
# GOOD - String reference prevents circular import
user = relationship("User", back_populates="bookings")
```

### ❌ DON'T: Import Models in Other Models

```python
# BAD - Will cause circular import error
from app.modules.auth.models import User  # ❌ Never in models.py
user = relationship(User, back_populates="bookings")
```

---

### ✅ DO: Inject Dependencies

```python
# GOOD - Dependencies injected
def get_my_bookings(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ...
```

### ❌ DON'T: Create Sessions Manually

```python
# BAD - Manual session creation
def get_my_bookings(current_user: User = Depends(get_current_user)):
    db = SessionLocal()  # ❌ Don't do this
    try:
        bookings = db.query(Booking).all()
    finally:
        db.close()
```

---

### ✅ DO: Use Separate Schemas

```python
# GOOD - Different schemas for different operations
class BookingCreate(BaseModel): ...  # For POST
class BookingUpdate(BaseModel): ...  # For PATCH
class BookingRead(BaseModel): ...    # For responses
```

### ❌ DON'T: Reuse One Schema Everywhere

```python
# BAD - One schema trying to do everything
class Booking(BaseModel):
    id: int | None = None  # ❌ Optional for create
    user_id: int | None = None  # ❌ Confusing
    ...
```

---

## Decision Flowchart

Use this to decide where your code should go:

```mermaid
flowchart TD
    Q1{What are you adding?}

    Q1 -->|New env variable| A1[core/config.py]
    Q1 -->|New exception type| A2[core/exceptions.py]
    Q1 -->|JWT/password logic| A3[core/security.py]
    Q1 -->|Logging setup| A10[core/logging.py]
    Q1 -->|New database table| A4[modules/DOMAIN/models.py]
    Q1 -->|API request/response shape| A5[modules/DOMAIN/schemas.py]
    Q1 -->|Business logic/queries| A6[modules/DOMAIN/service.py]
    Q1 -->|New API endpoint| A7[api/v1/endpoints/DOMAIN.py]
    Q1 -->|Shared auth dependency| A8[api/v1/dependencies.py]
    Q1 -->|Middleware/startup| A9[main.py]
    Q1 -->|Unit test for service| A11[tests/unit/test_modules/]
    Q1 -->|Integration test for endpoint| A12[tests/integration/api/v1/]

    A4 --> R1[Remember: Add to db/base.py!]
    A7 --> R2[Remember: Add to api/v1/router.py!]
```

---

## pyproject.toml Example

```toml
[project]
name = "wellnest-service"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.109.0",
    "uvicorn[standard]>=0.27.0",
    "sqlalchemy>=2.0.0",
    "alembic>=1.13.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "python-jose[cryptography]>=3.3.0",
    "passlib[bcrypt]>=1.7.4",
    "psycopg2-binary>=2.9.0",  # PostgreSQL
    "rich>=13.0.0",            # Beautiful logging
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "pytest-mock>=3.10.0",
    "pytest-cov>=4.0.0",
    "httpx>=0.26.0",
    "aiosqlite>=0.19.0",
    "ruff>=0.1.0",
    "pre-commit>=3.0.0",
]
```

---

## .env.example

```bash
# Application
APP_NAME="WellNest Service"
DEBUG=false
API_V1_PREFIX="/api/v1"

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/dbname
DB_ECHO=false

# Security
SECRET_KEY=your-super-secret-key-change-in-production
ACCESS_TOKEN_EXPIRE_MINUTES=30
ALGORITHM=HS256

# CORS (comma or JSON array)
CORS_ORIGINS=["http://localhost:3000","https://yourdomain.com"]

# Optional
REDIS_URL=redis://localhost:6379
SENTRY_DSN=
```


## Quick Start Checklist

When adding a **new module** (e.g., `payments`):

**In `modules/` (Business Logic):**
- [ ] Create `modules/payments/` folder
- [ ] Create `models.py` with table definition
- [ ] Create `schemas.py` with Create/Update/Read schemas
- [ ] Create `service.py` with CRUD class

**In `api/v1/` (HTTP Layer):**
- [ ] Create `api/v1/endpoints/payments.py` with routes
- [ ] Import and add router to `api/v1/router.py`

**Database:**
- [ ] Add model import to `db/base.py`
- [ ] Run `alembic revision --autogenerate -m "Add payments table"`
- [ ] Run `alembic upgrade head`

**Tests:**
- [ ] Create `tests/unit/test_modules/test_payments_service.py`
- [ ] Create `tests/integration/api/v1/test_payments_endpoints.py`
- [ ] Run `pytest -v` to verify

---
