from fastapi import FastAPI, UploadFile, File, HTTPException
import tempfile, shutil, os
from validator import process_image
from notify import update_airtable, send_email
from dotenv import load_dotenv
load_dotenv()


app = FastAPI(title="BookLeaf Cover Validator API")

@app.post("/analyze")
async def analyze_cover_api(file: UploadFile = File(...)):
    try:
        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        # Run core validator
        report = process_image(tmp_path)
        isbn = os.path.basename(file.filename).split("_")[0]

        # Compute status
        status = "PASS" if report["cover_valid"] else "REVIEW NEEDED"
        confidence = report.get("confidence_score", 0)
        overlay_url = report.get("overlay_path", "")
        message = report.get("validation_message", "")

        issues_list=[]
        if len(report.get("unauthorized_text_in_award_zone", []))>0:
            issues_list.append("Badge Overlap") 
        if len(report.get("text_in_safe_margin", []))>0:
            issues_list.append("Safe Margin") 
        if not issues_list:
            issues_list = ["No specific issues detected."]
        # Airtable update
        airtable_record = update_airtable(
            isbn=isbn,
            status=status,
            confidence=confidence,
            issues=issues_list,
            overlay_url=overlay_url,
            validation_message=message,
        ) # type: ignore

        # Email notify

        send_email(
            isbn=isbn,
            status=status,
            issues=airtable_record.get("fields", {}).get("Issue Type", []),
            overlay_url=overlay_url,
            confidence=confidence,
            to_email=airtable_record.get("fields", {}).get("Author Email", "ops@bookleaf.example"),
        ) # type: ignore

        return {
            "isbn": isbn,
            "status": status,
            "confidence": confidence,
            "validation_message": message,
            "airtable_record_id": airtable_record["id"],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))