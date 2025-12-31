# Backend Template Service

A production-ready FastAPI backend template using Hybrid Modular Architecture.

## Features

- **FastAPI**: High-performance async web framework.
- **SQLAlchemy (Async)**: Modern ORM with async support.
- **Alembic**: Database migrations.
- **Pydantic v2**: Data validation and settings.
- **Modular Architecture**: Clean separation of business logic and API layer.
- **Rich Logging**: Beautiful console logs.
- **Testing**: Pytest with async client and DB support.
- **Code Quality**: Ruff, Mypy, and pre-commit hooks.

## Getting Started

### Prerequisites

- [uv](https://github.com/astral-sh/uv) installed.

### Setup

1. Initialize environment:
   ```bash
   cp .env.example .env
   ```
2. Install dependencies:
   ```bash
   uv sync
   ```
3. Create alembic config:
   ```bash
   alembic init migrations
   ```
4. Start development server:
   ```bash
   uv run uvicorn app.main:app --reload
   ```
5. Run migrations (optional):
   ```bash
   uv run alembic upgrade head
   ```



### Testing

Run all tests:
   ```bash
   uv run pytest
   ```

### Code Quality

Run linters and formatting:
```bash
uv run pre-commit run --all-files
```

> FYI, the auth and users modules are for testing purposes only.
> ❗️ **Note**: Details on how to use the add modules and how to structure your project can be found in the `Guidelines` folder.
