from fastapi import FastAPI
from route import router

app = FastAPI(title="Mental Health AI", version="1.0")

app.include_router(router)
