from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers.rexomni_endpoints import router as rexomni_router
from app.routers.florence_endpoints import router as florence_router
#from app.health import router as health_router
from app.routers.jobs import router as jobs_router



app = FastAPI(
    title="Auto Labeling Service",
    description=(
        "Vision inference API for auto-labeling.\n\n"
        "• RexOmni tasks\n"
        "• Florence-2 vision-language tasks\n\n"
        "Model selection is handled at the application layer."
    ),
    version="2.0",
)

# -----------------------------
# CORS
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Routers
# -----------------------------
app.include_router(rexomni_router)
app.include_router(florence_router)
app.include_router(jobs_router)
#app.include_router(health_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=6996,
        reload=True,
    )
