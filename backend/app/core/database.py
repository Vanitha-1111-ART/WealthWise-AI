from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import declarative_base
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

# Base class for SQLAlchemy Models
Base = declarative_base()

# Global engine and sessionmaker references that can be overridden on fallback
engine = None
AsyncSessionLocal = None

def init_db_engine(database_url: str):
    """
    Initializes or overrides the global SQLAlchemy async engine and sessionmaker.
    Supports both PostgreSQL and SQLite fallback dynamics.
    """
    global engine, AsyncSessionLocal
    
    # Configure arguments dynamically based on database type
    if "sqlite" in database_url:
        connect_args = {"check_same_thread": False}
        engine_args = {}
    else:
        connect_args = {}
        engine_args = {
            "pool_size": 20,
            "max_overflow": 10,
            "pool_pre_ping": True
        }

    engine = create_async_engine(
        database_url,
        echo=False,
        connect_args=connect_args,
        **engine_args
    )
    
    AsyncSessionLocal = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False
    )
    logger.info(f"Database engine initialized for: {database_url.split('@')[-1] if '@' in database_url else database_url}")

# Initialize default engine from Settings (PostgreSQL async url)
init_db_engine(settings.DATABASE_URL)

async def get_db():
    """
    Dependency function to yield database sessions.
    Refers dynamically to the active sessionmaker local factory.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session error: {str(e)}")
            raise
        finally:
            await session.close()
