from fastapi import FastAPI

from smart_doc.app.routes import router


app = FastAPI()
app.include_router(router)

