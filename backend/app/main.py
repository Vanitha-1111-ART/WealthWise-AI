import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

from app.core.config import settings
from app.core.database import engine, Base
from app.api.endpoints import router as api_router

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI App
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Enterprise AI-Powered Digital Wealth Management Platform for IDBI Bank.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middlewares configuration (Allows frontend connection)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for hackathon simplicity
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Router
app.include_router(api_router)

# Database Tables Initialization on Startup
@app.on_event("startup")
async def startup_event():
    logger.info("Verifying database connection...")
    from app.core.database import engine as active_engine, Base, init_db_engine
    from sqlalchemy import text
    
    try:
        # Test the connection (lazy execution starts now)
        async with active_engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        logger.info("PostgreSQL connection verified successfully.")
    except Exception as e:
        logger.warning(f"PostgreSQL connection failed: {str(e)}")
        logger.warning("Falling back to local SQLite database ('./wealthwise.db') for zero-config execution.")
        
        # Re-initialize with SQLite async driver
        sqlite_url = "sqlite+aiosqlite:///./wealthwise.db"
        init_db_engine(sqlite_url)
        
    # Initialize database tables
    try:
        from app.core.database import engine as active_engine_now
        async with active_engine_now.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables initialized successfully.")
    except Exception as e:
        logger.error(f"Database table initialization failed: {str(e)}")
        logger.error("Please verify database configuration parameters.")

# Global Exception Handlers for Enterprise Robustness
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled Exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An internal server error occurred. Please contact WealthWise support.",
            "error_type": type(exc).__name__,
            "message": str(exc)
        }
    )

# Root Endpoint
@app.get("/")
async def root():
    return {
        "status": "online",
        "service": settings.PROJECT_NAME,
        "docs_url": "/docs",
        "api_version": "1.0.0"
    }

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
