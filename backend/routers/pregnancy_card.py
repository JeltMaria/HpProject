from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text, create_engine, select
from sqlalchemy.orm import sessionmaker
from pydantic import BaseModel
from typing import Dict, Any
from datetime import date
import json

router = APIRouter()
DATABASE_URL = "данные для подключение к Вашей БД"

def get_db():
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class PregnancyCardData(BaseModel):
    data: Dict[str, Any]

def validate_required_fields(data: Dict[str, Any]) -> None:
    required_fields = {
        "full_name": "ФИО пациентки",
        "date_of_birth": "Дата рождения",
        "menstruation_start_age": "Возраст начала менструации",
        "current_pregnancy_last_menstruation_start": "Дата начала последней менструации"
    }
    for field_id, field_name in required_fields.items():
        if field_id not in data or data[field_id] is None or data[field_id] == "":
            raise HTTPException(status_code=400, detail=f"Обязательное поле '{field_name}' не заполнено")

def validate_columns(db: Session, field_ids: list[str]) -> None:
    query = text("SELECT column_name FROM information_schema.columns WHERE table_schema = 'public' AND table_name = 'pregnancy_card'")
    valid_columns = {row[0] for row in db.execute(query).fetchall()}
    invalid_fields = [fid for fid in field_ids if fid not in valid_columns]
    if invalid_fields:
        raise HTTPException(status_code=400, detail=f"Следующие поля отсутствуют: {', '.join(invalid_fields)}")

def prepare_sql_value(value: Any, field_id: str) -> Any:
    if value is None:
        return None
    date_fields = ["date_of_birth", "current_pregnancy_last_menstruation_start", "current_pregnancy_last_menstruation_end", "maternity_leave_start_date"]
    if field_id in date_fields and value:
        try:
            return date.fromisoformat(value)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Неверный формат даты для поля {field_id}")
    jsonb_fields = ["past_diseases", "vaccinations", "gynecological_diseases", "previous_pregnancies_details", "blood_urine_tests", "ultrasound_data", "screening_results", "medications_supplements"]
    if field_id in jsonb_fields and isinstance(value, (list, dict)):
        return json.dumps(value)
    boolean_fields = ["hereditary_multiple_pregnancies", "menstruation_regular", "menstruation_pain", "current_pregnancy_regular_visits", "wasserman_test", "toxoplasmosis_test"]
    if field_id in boolean_fields:
        return value == "true" or value is True
    return value

@router.post("/create-pregnancy-card")
async def create_pregnancy_card(card_data: PregnancyCardData, db: Session = Depends(get_db)):
    try:
        validate_required_fields(card_data.data)
        validate_columns(db, card_data.data.keys())
        columns = [f'"{fid}"' for fid in card_data.data if card_data.data[fid] is not None and card_data.data[fid] != ""]
        values = [f':{fid}' for fid in card_data.data if card_data.data[fid] is not None and card_data.data[fid] != ""]
        params = {fid: prepare_sql_value(val, fid) for fid, val in card_data.data.items() if val is not None and val != ""}
        query = text(f"INSERT INTO pregnancy_card ({', '.join(columns)}) VALUES ({', '.join(values)}) RETURNING id")
        result = db.execute(query, params)
        new_card_id = result.fetchone()[0]
        db.commit()
        return {"message": "Карта беременности успешно создана", "card_id": new_card_id}
    except HTTPException as e:
        db.rollback()
        raise e
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Ошибка при создании карты: {str(e)}")

@router.get("/pregnancy-card/{user_id}")
async def get_pregnancy_card(user_id: int, db: Session = Depends(get_db)):
    query = text("SELECT * FROM pregnancy_card WHERE user_id = :user_id")
    result = db.execute(query, {"user_id": user_id}).fetchone()
    if not result:
        raise HTTPException(status_code=404, detail="Карта беременности не найдена")
    pregnancy_card = dict(result)
    jsonb_fields = ["past_diseases", "vaccinations", "gynecological_diseases", "previous_pregnancies_details", "blood_urine_tests", "ultrasound_data", "screening_results", "medications_supplements"]
    for field in jsonb_fields:
        if pregnancy_card.get(field):
            try:
                pregnancy_card[field] = json.loads(pregnancy_card[field]) if isinstance(pregnancy_card[field], str) else pregnancy_card[field]
            except json.JSONDecodeError:
                pregnancy_card[field] = []
        else:
            pregnancy_card[field] = []
    date_fields = ["date_of_birth", "current_pregnancy_last_menstruation_start", "current_pregnancy_last_menstruation_end", "maternity_leave_start_date", "created_at", "updated_at"]
    for field in date_fields:
        if pregnancy_card.get(field):
            pregnancy_card[field] = pregnancy_card[field].strftime("%Y-%m-%d")
    return pregnancy_card