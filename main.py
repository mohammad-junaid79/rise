from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# Initialize FastAPI app
app = FastAPI(title="FastAPI Lambda Backend")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://fastapi-render-deployment-3.onrender.com/,*"],  # For production, restrict to CloudFront/API Gateway URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store (resets on each cold start)
users = []

# Pydantic model for request body
class User(BaseModel):
    name: str

# Routes
@app.get("/")
async def root():
    return {"message": "FastAPI Lambda Backend is running!"}

@app.post("/users")
async def create_user(user: User):   # use async for Lambda safety
    users.append(user.name)
    return {"message": f"User '{user.name}' created", "total_users": len(users)}

@app.get("/users/count")
async def count_users():
    return {"total_users": len(users)}



