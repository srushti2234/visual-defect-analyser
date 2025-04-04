from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "postgresql://postgres:postgre@localhost:5432/damage_db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Define the DamageAnalysis table
class DamageAnalysis(Base):
    __tablename__ = "damage_analysis"
    
    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String, index=True)
    damage_percentage = Column(Float)
    threshold = Column(Float)

# Create tables if they don't exist
Base.metadata.create_all(bind=engine)
