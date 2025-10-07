from typing import AsyncGenerator
from fastapi import HTTPException, Security, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.settings import settings

security = HTTPBearer()


async def api_key_auth(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Validate API key from Authorization header."""
    if credentials.credentials != settings.API_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials


async def get_db_session(request: Request) -> AsyncGenerator[AsyncSession, None]:
    """Get database session for each request."""
    async with AsyncSession(request.app.state.engine, expire_on_commit=False) as session:
        yield session