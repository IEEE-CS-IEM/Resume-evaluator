from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from datetime import datetime,timezone
from models import KYCRequest
from db import db
from fastapi import Path,APIRouter, HTTPException

router = APIRouter()

@router.post("/kyc",tags=["KYC"], summary="Submit KYC data")
def submit_kyc(data: KYCRequest):
    kyc_collection = db["kyc"]

    if kyc_collection.find_one({"email": data.email}):
        raise HTTPException(status_code=409, detail="KYC already submitted for this email.")

    safe_data = jsonable_encoder(data)
    safe_data["submitted_at"] = datetime.now(timezone.utc)
    kyc_collection.insert_one(safe_data)

    return {"message": "KYC data submitted successfully!"}

@router.get("/kyc/{email}",tags=["KYC"], summary="Fetch KYC data by email")
def get_kyc(email: str = Path(..., description="User's email address")):
    kyc_collection = db["kyc"]
    user_data = kyc_collection.find_one({"email": email})

    if not user_data:
        raise HTTPException(status_code=404, detail="No KYC data found for this email.")
    user_data.pop("_id", None)  # Optional: hide MongoDB's ObjectId
    return user_data

@router.put("/kyc/{email}", tags=["KYC"])
def update_kyc(email: str, updated_data: KYCRequest):
    kyc_collection = db["kyc"]
    if not kyc_collection.find_one({"email": email}):
        raise HTTPException(status_code=404, detail="KYC record not found.")

    result = kyc_collection.update_one(
        {"email": email},
        {"$set": jsonable_encoder(updated_data)}
    )

    return {"message": "KYC data updated successfully."}

@router.delete("/kyc/{email}", tags=["KYC"])
def delete_kyc(email: str):
    kyc_collection = db["kyc"]
    result = kyc_collection.delete_one({"email": email})

    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="No KYC record found to delete.")

    return {"message": "KYC data deleted successfully."}
