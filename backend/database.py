from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

DATABASE_URL = "sqlite:///./gridguard.db"
Base = declarative_base()

class Meter(Base):
    __tablename__ = "meters"
    id = Column(Integer, primary_key=True, index=True)
    meter_id = Column(String, unique=True, index=True)
    location = Column(String, nullable=True)

class Detection(Base):
    __tablename__ = "detections"
    id = Column(Integer, primary_key=True, index=True)
    meter_id = Column(String, ForeignKey("meters.meter_id"))
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    is_theft = Column(Boolean)
    confidence = Column(Float)
    raw_data_path = Column(String, nullable=True) # Path to the JSON snapshot of raw readings

# Setup DB
from sqlalchemy import create_engine
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
