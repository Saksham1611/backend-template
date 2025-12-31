import logging
from collections.abc import AsyncGenerator

from sqlalchemy import MetaData, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.ext.declarative import declarative_base

from app.core.config import settings

logger = logging.getLogger(__name__)

engine = create_async_engine(settings.DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

# Create declarative base
Base = declarative_base()

# Metadata for migrations
metadata = MetaData()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db() -> None:
    """Initialize database connection and optionally create tables based on environment."""
    try:
        # Test database connection first
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))

        # depending on environment, create tables if needed
        if settings.ENVIRONMENT == "dev":
            if settings.CREATE_TABLES_ON_STARTUP:
                async with engine.begin() as conn:
                    await conn.run_sync(Base.metadata.create_all)
                logger.info(
                    "Development: Database tables created/verified successfully"
                )
            else:
                logger.info(
                    "Development: Database connection established (table creation skipped)"
                )

        # In production, we assume tables are managed by migrations
        elif settings.ENVIRONMENT == "prod":
            logger.info("Production: Database connection established")

        # For testing, always create tables
        elif settings.ENVIRONMENT == "local":
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created for local")

        # Unknown environment - be conservative
        else:
            logger.info(
                f"Unknown environment '{settings.ENVIRONMENT}': Database connection established"
            )

    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        logger.error(
            f"Database URL: {settings.DATABASE_URL.split('@')[1] if '@' in settings.DATABASE_URL else 'Invalid URL'}"
        )
        raise


async def close_db() -> None:
    """Close database connection gracefully."""
    try:
        await engine.dispose()
        logger.info("Database connection closed successfully")
    except Exception as e:
        logger.error(f"Error closing database connection: {e}")


async def health_check_db() -> bool:
    """Check if database is accessible."""
    try:
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False
