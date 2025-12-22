# AGENTS.md - The Holy Grail for AI-Assisted Development

This file defines the structural and coding standards for this project. All AI agents and developers must adhere to these rules when creating or modifying files.

## Project Structure

```
backend-template/
├── pyproject.toml              # Modern dependency management (uv)
├── .pre-commit-config.yaml
├── .env.example
├── alembic.ini                 # DB Migrations config
├── app/
│   ├── main.py                 # Entry point + Lifespan events
│   ├── core/                   # Cross-cutting concerns
│   │   ├── config.py           # Pydantic Settings
│   │   ├── exceptions.py       # Custom HTTP exceptions
│   │   ├── logging.py          # Rich logging setup
│   │   └── security.py         # JWT + password hashing
│   ├── db/                     # Database layer
│   │   ├── session.py          # Engine & SessionLocal
│   │   └── base.py             # The "Registry" file (imports all models)
│   ├── api/                    # API Layer
│   │   └── v1/
│   │       ├── router.py       # Switchboard - aggregates endpoints
│   │       ├── dependencies.py # Bouncers (get_db, get_current_user)
│   │       └── endpoints/      # API route handlers
│   └── modules/                # PURE BUSINESS LOGIC (No API code!)
│       ├── <domain>/
│       │   ├── models.py       # Domain models
│       │   ├── schemas.py      # Pydantic schemas
│       │   └── service.py      # Business logic
├── tests/                      # Testing framework
│   ├── conftest.py             # Shared fixtures
│   ├── unit/                   # Unit tests
│   └── integration/            # Integration tests
└── migrations/                 # Alembic migrations
```

## How to Add New Logic/Files

1.  **New Domain Entity**:
    - Create a folder in `app/modules/<domain>/`.
    - Define models in `models.py`.
    - Register models in `app/db/base.py`.
    - Define schemas in `schemas.py`.
    - Implement logic in `service.py`.
2.  **New API Endpoints**:
    - Create a router file in `app/api/v1/endpoints/<domain>.py`.
    - Register the router in `app/api/v1/router.py`.
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
