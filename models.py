from pydantic import BaseModel, EmailStr, HttpUrl, Field
from typing import Optional

class KYCRequest(BaseModel):
    full_name: str
    email: EmailStr
    phone: str = Field(..., pattern=r'^\d{10}$')
    address: Optional[str] = None
    linkedin: Optional[HttpUrl] = None
    github: Optional[HttpUrl] = None