from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from smart_doc.app.routes import router


app = FastAPI()

# ... router setup ...
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows requests from your React Frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)

