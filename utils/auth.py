import os
import jwt
from datetime import datetime, timedelta
from functools import wraps

# In a production environment, use a secure secret key stored in environment variables
SECRET_KEY = "your-secret-key-here"  # Change this in production
ADMIN_PASSWORD = "admin123"  # Change this in production

def create_token():
    """Create a JWT token for admin authentication"""
    expiration = datetime.utcnow() + timedelta(hours=24)
    return jwt.encode(
        {"exp": expiration},
        SECRET_KEY,
        algorithm="HS256"
    )

def verify_token(token):
    """Verify the JWT token"""
    try:
        jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return True
    except:
        return False

def check_password(password):
    """Check if the provided password matches the admin password"""
    return password == ADMIN_PASSWORD
