import os
from sqlalchemy import create_engine, Column, Integer, String, Date, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import pandas as pd

Base = declarative_base()

class LotteryRecord(Base):
    __tablename__ = 'lottery_records'
    
    id = Column(Integer, primary_key=True)
    period = Column(String(20), unique=True, index=True)
    date = Column(String(20))
    front_1 = Column(Integer)
    front_2 = Column(Integer)
    front_3 = Column(Integer)
    front_4 = Column(Integer)
    front_5 = Column(Integer)
    back_1 = Column(Integer)
    back_2 = Column(Integer)

class Database:
    def __init__(self):
        db_url = os.environ.get('DATABASE_URL')
        if not db_url:
            db_url = 'sqlite:///lottery.db'
        
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
    
    def insert_data_from_csv(self, csv_file):
        df = pd.read_csv(csv_file)
        
        for _, row in df.iterrows():
            existing = self.session.query(LotteryRecord).filter_by(period=row['period']).first()
            if not existing:
                record = LotteryRecord(
                    period=row['period'],
                    date=row['date'],
                    front_1=row['front_1'],
                    front_2=row['front_2'],
                    front_3=row['front_3'],
                    front_4=row['front_4'],
                    front_5=row['front_5'],
                    back_1=row['back_1'],
                    back_2=row['back_2']
                )
                self.session.add(record)
        
        self.session.commit()
        print("数据已导入数据库")
    
    def get_all_records(self):
        records = self.session.query(LotteryRecord).order_by(LotteryRecord.period.desc()).all()
        data = []
        for r in records:
            data.append({
                'period': r.period,
                'date': r.date,
                'front_1': r.front_1,
                'front_2': r.front_2,
                'front_3': r.front_3,
                'front_4': r.front_4,
                'front_5': r.front_5,
                'back_1': r.back_1,
                'back_2': r.back_2
            })
        return pd.DataFrame(data)

if __name__ == "__main__":
    db = Database()
    print("数据库初始化成功")
