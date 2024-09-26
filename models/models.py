from database.database import Base
from sqlalchemy import Column, Integer, String, Double

class pacientes(Base):
    __tablename__ = "pacientes"

    idpacientes = Column(Integer, primary_key=True, nullable=False)
    nombre = Column(String, nullable=False)
    apellido = Column(String, nullable=False)
    pregnancies = Column(Integer, nullable=False)
    glucose = Column(Integer, nullable=False)
    bloodpressure = Column(Integer, nullable=False)
    skinthickness = Column(Integer, nullable=False)
    insulin = Column(Integer, nullable=False)
    BMI = Column(Double, nullable=False)
    diabetespedigreefunction =Column(Double, nullable=False)
    age = Column(Integer, nullable=False)