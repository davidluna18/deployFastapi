import pickle
from fastapi import APIRouter, Depends
import numpy as np

from database.database import SessionLocal, get_db
from schemas import schemas
from models import models
from models.models import pacientes

router = APIRouter()


pkl_filename = "RFDiabetesv102.pkl"
with open(pkl_filename, 'rb') as file:
    model = pickle.load(file)

labels = ['Sano','Posible diabetes']  

@router.get("/")
async def root():
    return {"message": "Modelos predictivos"}


@router.post('/predict')
def predict_diabetes(data:schemas.paciente, db: SessionLocal = Depends(get_db)):

    data = data.model_dump()
    
    idpaciente=data['idpaciente']
    
    paciente = db.query(models.pacientes).filter(
                                pacientes.idpacientes==idpaciente,
                            ).first()
    
    print(paciente.nombre)
    xin = np.array([paciente.pregnancies,
                    paciente.glucose,
                    paciente.bloodpressure,
                    paciente.skinthickness,
                    paciente.insulin,
                    paciente.BMI,
                    paciente.diabetespedigreefunction,
                    paciente.age]).reshape(1,8)

    prediction = model.predict(xin)
    yout = labels[prediction[0]]
    
    return {
        'name': paciente.nombre,
        'prediction': yout
    }