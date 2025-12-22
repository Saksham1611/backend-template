# Import the Base
from app.db.session import Base  # noqa

# Import ALL models here so Alembic can "see" them
from app.modules.users.models import User  # noqa
