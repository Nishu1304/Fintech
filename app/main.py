from fastapi import FastAPI
from app.routers import preprocessing, pattern_prediction, login_signup, anamoly_detection
from app import config

app = FastAPI()


config.add_cors_middleware(app)

app.include_router(preprocessing.router, prefix="/api")
app.include_router(login_signup.app, prefix="/api/users")
app.include_router(anamoly_detection.router, prefix="/api/anomalies")
app.include_router(pattern_prediction.router, prefix="/api/patterns")