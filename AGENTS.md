# AGENTS.md - The Holy Grail for AI-Assisted Development

This file defines the structural and coding standards for this project. All AI agents and developers must adhere to these rules when creating or modifying files.

## Project Structure

```
wellnest-activity-engine/
├── pyproject.toml              # Modern dependency management (uv)
├── .pre-commit-config.yaml
├── .env.example
├── alembic.ini                 # DB Migrations config (points to app/db)
├── app/
│   ├── main.py                 # Entry point + Lifespan events
│   ├── core/                   # Cross-cutting concerns
│   │   ├── config.py           # Pydantic Settings
│   │   ├── exceptions.py       # Custom HTTP exceptions
│   │   ├── logging.py          # Rich logging setup
│   │   ├── security.py         # JWT + password hashing
│   │   └── manifest.py         # Multimodal Manifest Pydantic models
│   ├── db/                     # Database layer
│   │   └── database.py         # Engine, AsyncSessionLocal, and Base
│   ├── api/                    # API Layer
│   │   ├── routes/             # API route handlers
│   │   │   ├── __init__.py     # Aggregates routes into api_router
│   │   │   ├── auth.py
│   │   │   └── users.py
│   │   └── dependencies.py     # Auth & Context Bouncers (get_db, get_current_user)
│   ├── services/               # Business logic / Orchestration
│   │   ├── auth_service.py
│   │   └── user_service.py
│   ├── repositories/           # Data Access Layer (Direct DB queries)
│   ├── models/                 # SQLAlchemy ORM Models
│   │   └── user.py
│   └── schemas/                # Shared Pydantic Schemas
│       ├── auth.py
│       └── user.py
├── tests/                      # Testing framework
│   ├── conftest.py             # Shared fixtures & DB overrides
│   ├── unit/                   # Unit tests (Services & Logic)
│   └── integration/            # Integration tests (API Endpoints)
└── migrations/                 # Alembic migrations
```

## How to Add New Logic/Files

1.  **New Domain Logic**:
    - Define the model in `app/models/<domain>.py`.
    - Create relevant schemas in `app/schemas/<domain>.py`.
    - Implement persistence logic in `app/repositories/<domain>_repo.py`.
    - Implement business orchestration in `app/services/<domain>_service.py`.
2.  **New API Endpoints**:
    - Create a route file in `app/api/routes/<domain>.py`.
    - Register the router in `app/api/routes/__init__.py`.
3.  **New Configuration**:
    - Add to `app/core/config.py`.

## Coding Guidelines (Python 3.12+)

### match-case Syntax
Prefer `match-case` over `if/elif/else` for pattern matching.
```python
match status:
    case "active":
        ...
    case "inactive":
        ...
    case _:
        ...
```

### Never Nester
Avoid deep nesting. Use early returns and guard clauses.
```python
def process_data(data):
    if not data:
        return None
    # process...
```

### Modern Type Hints
Use built-in generics and the union pipe operator.
```python
def get_items(ids: list[int]) -> dict[int, str] | None:
    ...
```

### Pydantic-First Parsing
Use Pydantic v2's native validation (`model_validate`, `ConfigDict`). Avoid manual attribute checking.

### Pathlib
Use `pathlib.Path` for file system operations.

### uv for All Commands
Always run commands using `uv`.
- `uv add <package>`
- `uv run <command>`
- `uv run pytest`

### Error Handling
Use custom exceptions from `app.core.exceptions`.

### No Inline Ignores
Fix types at the source instead of using `# type: ignore`.


### Run the application
- `fastapi dev --reload --reload-exclude "test.db" --reload-exclude ".pytest_cache"`  # development
- `fastapi run`           # for production
