# Backend Template Setup Guide

> A comprehensive guide to creating a production-ready FastAPI backend using the **Hybrid Modular Architecture** (Centralized API + Domain Modules).

## Tech Stack Overview

| Tool | Purpose |
|------|---------|
| **uv** | Modern Python dependency management |
| **FastAPI** | High-performance async web framework |
| **SQLAlchemy** | Async ORM with PostgreSQL/SQLite |
| **Alembic** | Database migrations |
| **Rich** | Beautiful terminal logging |
| **Pydantic** | Data validation and settings |
| **Pytest** | Testing framework |
| **Ruff** | Linting and formatting |
| **Mypy** | Static type checking |

---

## 🚀 Phase 1: Project Initialization

Create the project shell and install the modern stack.

### Step 1.1: Initialize Project with UV

```bash
# Initialize the project
uv init wellnest-template
cd wellnest-template
```

### Step 1.2: Install Core Dependencies

```bash
uv add fastapi "uvicorn[standard]" sqlalchemy asyncpg pydantic-settings alembic rich python-jose[cryptography] passlib[bcrypt] python-multipart
```

### Step 1.3: Install Dev/Test Dependencies

```bash
uv add --dev pytest pytest-asyncio httpx pytest-mock pre-commit ruff
```

### Step 1.4: Create Directory Structure

```bash
# Create the full project structure
mkdir -p app/api/v1/endpoints
mkdir -p app/core
mkdir -p app/db
mkdir -p app/modules/auth
mkdir -p app/modules/users
mkdir -p tests/unit
mkdir -p tests/integration
```

### Final Structure

```
wellnest-template/
├── app/
│   ├── api/
│   │   └── v1/
│   │       ├── endpoints/
│   │       │   └── users.py
│   │       ├── dependencies.py
│   │       └── router.py
│   ├── core/
│   │   ├── config.py
│   │   └── logging.py
│   ├── db/
│   │   ├── base.py
│   │   └── session.py
│   ├── modules/
│   │   ├── auth/
│   │   └── users/
│   │       ├── schemas.py
│   │       └── service.py
│   └── main.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── conftest.py
├── alembic/
├── .env
├── .pre-commit-config.yaml
└── pyproject.toml
```

---

## 🛠️ Phase 2: Core Configuration

Set up the "nervous system" of the app: Settings and Logging.

### Step 2.1: Create `app/core/config.py`

> Pydantic Settings for type-safe configuration management.

```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal

class Settings(BaseSettings):
    PROJECT_NAME: str = "Wellnest Template Service"
    ENVIRONMENT: Literal["local", "dev", "prod"] = "local"

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://user:password@localhost/dbname"

    # Security
    SECRET_KEY: str = "change_this_in_production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True)

settings = Settings()
```

### Step 2.2: Create `app/core/logging.py`

> Rich integration for beautiful, readable logs.

```python
import logging
from rich.logging import RichHandler

def setup_logging(level: str = "INFO"):
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.handlers = []  # Clear existing handlers

    rich_handler = RichHandler(
        rich_tracebacks=True,
        markup=True,
        show_time=True,
        show_path=False
    )
    rich_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(rich_handler)

    # Quiet down noisy libs
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
```

---

## 🗄️ Phase 3: Database & Registry

This is the "Hybrid" trick to make Alembic work with modular files.

### Step 3.1: Create `app/db/session.py`

```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from app.core.config import settings

engine = create_async_engine(settings.DATABASE_URL, echo=False)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)

class Base(DeclarativeBase):
    pass
```

### Step 3.2: Create `app/db/base.py` (The Registry)

> [!IMPORTANT]
> This file is crucial for Alembic. Import ALL models here so migrations can detect them.

```python
# Import the Base
from app.db.session import Base

# Import ALL models here so Alembic can "see" them
# Example: from app.modules.auth.models import User  # noqa
```

---

## 🧱 Phase 4: Business Modules

Create domain modules to encapsulate business logic. Each module contains:
- `schemas.py` - Pydantic models for validation
- `service.py` - Business logic (no API code)
- `models.py` - SQLAlchemy models (optional)

### Step 4.1: Create `app/modules/users/schemas.py`

```python
from pydantic import BaseModel, EmailStr

class UserBase(BaseModel):
    email: EmailStr

class UserCreate(UserBase):
    password: str

class UserResponse(UserBase):
    id: int
    is_active: bool

    class Config:
        from_attributes = True
```

### Step 4.2: Create `app/modules/users/service.py`

> Pure business logic - no API/HTTP code here.

```python
class UserService:
    async def create_user(self, data: dict):
        # Business logic simulation
        return {"id": 1, "email": data["email"], "is_active": True}
```

---

## 🔌 Phase 5: The API Layer

Wire the business logic to HTTP endpoints using FastAPI Dependencies.

### Step 5.1: Create `app/api/v1/dependencies.py`

```python
from typing import Generator
from app.db.session import SessionLocal

async def get_db() -> Generator:
    async with SessionLocal() as session:
        yield session
```

### Step 5.2: Create `app/api/v1/endpoints/users.py`

```python
from fastapi import APIRouter, Depends
from app.modules.users.schemas import UserCreate, UserResponse
from app.modules.users.service import UserService

router = APIRouter()
service = UserService()  # In real app, inject this

@router.post("/", response_model=UserResponse)
async def create_user(user_in: UserCreate):
    return await service.create_user(user_in.model_dump())
```

### Step 5.3: Create `app/api/v1/router.py` (The Switchboard)

```python
from fastapi import APIRouter
from app.api.v1.endpoints import users

api_router = APIRouter()
api_router.include_router(users.router, prefix="/users", tags=["Users"])
```

---

## 🏁 Phase 6: Main Entrypoint

Connect the brain (`main.py`) to logging and router.

### Step 6.1: Create `app/main.py`

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.core.config import settings
from app.core.logging import setup_logging
from app.api.v1.router import api_router
import logging

setup_logging()
logger = logging.getLogger("app")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"[bold green]Starting {settings.PROJECT_NAME}...[/]", extra={"markup": True})
    yield
    logger.info("[bold red]Shutting down...[/]", extra={"markup": True})

app = FastAPI(
    title=settings.PROJECT_NAME,
    lifespan=lifespan
)

app.include_router(api_router, prefix="/api/v1")

@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "0.1.0"}
```

---

## 🐘 Phase 7: Alembic Setup (Migrations)

Configure Alembic for async SQLAlchemy migrations.

### Step 7.1: Initialize Alembic

```bash
uv run alembic init -t async alembic
```

### Step 7.2: Edit `alembic/env.py`

Find and modify these sections:

```python
# alembic/env.py

# ... imports ...
from app.core.config import settings
from app.db.base import Base  # <--- IMPORT THE REGISTRY

# ... code ...
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)  # <--- USE SETTINGS

target_metadata = Base.metadata  # <--- SET METADATA
```

### Step 7.3: Create Your First Migration

```bash
# After creating models, generate migration
uv run alembic revision --autogenerate -m "initial"

# Apply migration
uv run alembic upgrade head
```

---

## 🧪 Phase 8: Testing Setup

### Step 8.1: Create `tests/conftest.py`

```python
import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app

@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
```

### Step 8.2: Example Test Structure

**Unit Test** (`tests/unit/test_user_service.py`):
```python
import pytest
from app.modules.users.service import UserService

@pytest.mark.asyncio
async def test_create_user():
    service = UserService()
    result = await service.create_user({"email": "test@example.com", "password": "secret"})
    assert result["email"] == "test@example.com"
    assert result["is_active"] is True
```

**Integration Test** (`tests/integration/test_users_api.py`):
```python
import pytest

@pytest.mark.asyncio
async def test_create_user_endpoint(client):
    response = await client.post("/api/v1/users/", json={
        "email": "test@example.com",
        "password": "secret123"
    })
    assert response.status_code == 200
    assert response.json()["email"] == "test@example.com"
```

---

## ✅ Phase 9: Environment & Running

### Step 9.1: Create `.env` File

```env
DATABASE_URL=sqlite+aiosqlite:///./dev.db
SECRET_KEY=dev_secret
ENVIRONMENT=local
```

### Step 9.2: Run the Server

```bash
uv run uvicorn app.main:app --reload
```

### Expected Output

You should see Rich formatted logs:
```
Starting Wellnest Template Service...
```

### Step 9.3: Verify

- **API Docs**: http://127.0.0.1:8000/docs
- **Health Check**: http://127.0.0.1:8000/health

---

## 🔒 Phase 10: Pre-commit & Code Quality

### Step 10.1: Create `.pre-commit-config.yaml`

```yaml
# .pre-commit-config.yaml
repos:
  # 1. Standard File Hygiene
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict

  # 2. Ruff (Linter & Formatter) - Replaces Black, Isort, Flake8
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.3.4
    hooks:
      - id: ruff
        args: [ --fix ]
      - id: ruff-format

  # 3. Mypy (Static Type Checker)
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.9.0
    hooks:
      - id: mypy
        additional_dependencies: [
          "pydantic>=2.0.0",
          "sqlalchemy[asyncio]>=2.0.0",
          "types-all"
        ]
        args: ["--ignore-missing-imports", "--disallow-untyped-defs"]

  # 4. Secret Scanning (Security)
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.4.0
    hooks:
      - id: detect-secrets
        args: ['--baseline', '.secrets.baseline']
        exclude: package.lock.json|uv.lock
```

### Step 10.2: Add Tool Configuration to `pyproject.toml`

```toml
[tool.ruff]
line-length = 88
target-version = "py311"

[tool.ruff.lint]
select = [
    "E",  # pycodestyle errors
    "W",  # pycodestyle warnings
    "F",  # pyflakes
    "I",  # isort (imports)
    "B",  # flake8-bugbear
    "C4", # flake8-comprehensions
    "UP", # pyupgrade (modernize python syntax)
]
ignore = [
    "E501",  # line too long, handled by black
    "B008",  # do not perform function calls in argument defaults (FastAPI relies on this)
]

[tool.mypy]
python_version = "3.11"
plugins = ["pydantic.mypy", "sqlalchemy.ext.mypy.plugin"]
ignore_missing_imports = true
check_untyped_defs = true

[tool.pytest.ini_options]
asyncio_mode = "auto"
python_files = "test_*.py"
```

### Step 10.3: Install & Configure Pre-commit

```bash
# Ensure pre-commit is installed
uv add --dev pre-commit

# Install the git hooks
uv run pre-commit install

# Establish secrets baseline
uv run detect-secrets scan > .secrets.baseline

# Run on all files (initial cleanup)
uv run pre-commit run --all-files
```

---

## 🧠 Architecture Summary

| Layer | Location | Responsibility |
|-------|----------|----------------|
| **API** | `app/api/v1/` | HTTP routing, request/response handling |
| **Core** | `app/core/` | Config, logging, shared utilities |
| **DB** | `app/db/` | Database session, model registry |
| **Modules** | `app/modules/` | Domain logic (schemas, services, models) |
| **Tests** | `tests/` | Unit and integration tests |

### Key Principles

1. **Separation of Concerns**: Business logic in `modules/`, HTTP wiring in `api/`
2. **Dependency Injection**: Use FastAPI's `Depends()` for services
3. **Type Safety**: Pydantic for validation, Mypy for static checking
4. **Registry Pattern**: Single `db/base.py` imports all models for Alembic

---

## 📋 Quick Reference Commands

| Command | Purpose |
|---------|---------|
| `uv run uvicorn app.main:app --reload` | Start dev server |
| `uv run pytest` | Run all tests |
| `uv run pytest tests/unit` | Run unit tests only |
| `uv run alembic revision --autogenerate -m "msg"` | Create migration |
| `uv run alembic upgrade head` | Apply migrations |
| `uv run pre-commit run --all-files` | Run all linters |
| `uv run ruff check . --fix` | Fix linting issues |
| `uv run ruff format .` | Format code |

---

## Related Notes

- [[Instructions Backend]] - Detailed backend development instructions
- [[Pytest Testing Guide]] - Comprehensive testing documentation
