from fastapi import FastAPI
from kyc_routes import router as kyc_router

app = FastAPI(
    title="Resume Evaluator Backend",
    description="A FastAPI-powered backend for KYC submission and resume-job matching.",
    version="1.0.0",
    contact={
        "name": "Archishman Das",
        "email": "archi.das15@gmail.com",
    },
)

app.include_router(kyc_router)
