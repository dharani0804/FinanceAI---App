from sqlalchemy import Column, Integer, String, Float, Date, DateTime, Boolean
from sqlalchemy.sql import func
from database import Base

class Transaction(Base):
    __tablename__ = "transactions"
    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date)
    description = Column(String)
    original_description = Column(String)
    amount = Column(Float)
    currency = Column(String, default="USD")
    category = Column(String, default="Uncategorized")
    bank_source = Column(String, default="Unknown Bank")
    is_edited = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.now())

class Goal(Base):
    __tablename__ = "goals"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String)
    target_amount = Column(Float)
    current_amount = Column(Float, default=0)
    currency = Column(String, default="USD")
    deadline = Column(Date)
    created_at = Column(DateTime, default=func.now())

class UserProfile(Base):
    __tablename__ = "user_profile"
    id = Column(Integer, primary_key=True)
    monthly_income = Column(Float, default=0)
    monthly_budget = Column(Float, default=0)
    savings_goal = Column(Float, default=0)
    currency = Column(String, default="USD")
    pay_frequency = Column(String, default="monthly")
    income_sources = Column(String, default="")
    created_at = Column(DateTime, default=func.now())
