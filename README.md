# Backend Template Service

A production-ready FastAPI backend template using Hybrid Modular Architecture.

## Features

- **FastAPI**: High-performance async web framework.
- **SQLAlchemy (Async)**: Modern ORM with async support.
- **Alembic**: Database migrations.
- **Pydantic v2**: Data validation and settings.
- **Simplified Architecture**: Flattened project structure for faster development and easier maintenance.
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
   alembic init alembic
   ```
4. Start development server:
   ```bash
   fastapi dev --reload --reload-exclude "test.db" --reload-exclude ".pytest_cache"
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
