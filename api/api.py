from fastapi import FastAPI

from .routers import slicing, inference, reconstruct, imgshow

app = FastAPI()

app.include_router(slicing.router)
app.include_router(inference.router)
app.include_router(reconstruct.router)
app.include_router(imgshow.router)


@app.get("/")
async def health_check():
    return {"status": "API is running"}
