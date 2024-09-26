from pydantic import BaseModel

class Diabetesdata(BaseModel):
    Pregnancies: float 
    Glucose: float 	
    BloodPressure: float 	
    SkinThickness: float 	
    Insulin: float 	
    BMI: float 
    DiabetesPedigreeFunction: float 	
    Age: float 