from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.core.settings import settings

security = HTTPBearer()


async def api_key_auth(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Validate API key from Authorization header."""
    if credentials.credentials != settings.API_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials