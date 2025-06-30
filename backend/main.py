from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import auth, child_image, pregnancy_card, dialog

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/api")
app.include_router(child_image.router, prefix="/api")
app.include_router(pregnancy_card.router, prefix="/api")
app.include_router(dialog.router, prefix="/api")

@app.get("/")
def read_root():
    return {"message": "HpProject API"}