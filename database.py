import os
from sqlalchemy import create_engine, Column, Integer, String, Date, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import NullPool
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

class PredictionRecord(Base):
    __tablename__ = 'prediction_records'
    
    id = Column(Integer, primary_key=True)
    predict_time = Column(String(30))
    target_period = Column(String(20), index=True)
    play_type = Column(String(50))
    
    front_numbers = Column(String(100))
    back_numbers = Column(String(50))
    
    front_probabilities = Column(String(200))
    back_probabilities = Column(String(100))
    
    actual_front = Column(String(100))
    actual_back = Column(String(50))
    
    is_matched = Column(Integer, default=0)
    match_level = Column(String(20))
    front_match_count = Column(Integer, default=0)
    back_match_count = Column(Integer, default=0)

class Database:
    def __init__(self):
        db_url = os.environ.get('DATABASE_URL')
        if not db_url:
            db_url = 'sqlite:///lottery.db'
        
        # 使用NullPool避免连接池问题
        self.engine = create_engine(db_url, poolclass=NullPool, connect_args={'connect_timeout': 10})
        Base.metadata.create_all(self.engine)
        
        # 使用scoped_session确保线程安全
        session_factory = sessionmaker(bind=self.engine)
        self.Session = scoped_session(session_factory)
    
    def get_session(self):
        """获取一个新的session"""
        return self.Session()
    
    def insert_data_from_csv(self, csv_file):
        df = pd.read_csv(csv_file)
        session = self.get_session()
        
        try:
            added_count = 0
            for _, row in df.iterrows():
                period_str = str(int(row['period']))
                
                existing = session.query(LotteryRecord).filter(
                    LotteryRecord.period == period_str
                ).first()
                
                if not existing:
                    record = LotteryRecord(
                        period=period_str,
                        date=str(row['date']),
                        front_1=int(row['front_1']),
                        front_2=int(row['front_2']),
                        front_3=int(row['front_3']),
                        front_4=int(row['front_4']),
                        front_5=int(row['front_5']),
                        back_1=int(row['back_1']),
                        back_2=int(row['back_2'])
                    )
                    session.add(record)
                    added_count += 1
                    
                    if added_count % 10 == 0:
                        session.commit()
            
            session.commit()
            print(f"数据已导入数据库，新增 {added_count} 条记录")
            return added_count
        except Exception as e:
            session.rollback()
            print(f"数据导入出错: {str(e)}")
            raise
        finally:
            session.close()
    
    def get_all_records(self):
        session = self.get_session()
        try:
            from sqlalchemy import cast, Integer
            records = session.query(LotteryRecord).order_by(cast(LotteryRecord.period, Integer).desc()).all()
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
        except Exception as e:
            session.rollback()
            print(f"查询数据出错: {str(e)}")
            raise
        finally:
            session.close()
    
    def get_latest_period(self):
        """获取数据库中最新的期号"""
        session = self.get_session()
        try:
            from sqlalchemy import cast, Integer
            latest = session.query(LotteryRecord).order_by(cast(LotteryRecord.period, Integer).desc()).first()
            return latest.period if latest else None
        except Exception as e:
            session.rollback()
            print(f"查询最新期号出错: {str(e)}")
            return None
        finally:
            session.close()
    
    def get_record_count(self):
        """获取数据库中的记录总数"""
        session = self.get_session()
        try:
            return session.query(LotteryRecord).count()
        except Exception as e:
            session.rollback()
            print(f"查询记录数出错: {str(e)}")
            return 0
        finally:
            session.close()
    
    def save_prediction(self, prediction_data):
        """保存预测记录"""
        session = self.get_session()
        try:
            record = PredictionRecord(
                predict_time=prediction_data['predict_time'],
                target_period=prediction_data['target_period'],
                play_type=prediction_data['play_type'],
                front_numbers=prediction_data['front_numbers'],
                back_numbers=prediction_data['back_numbers'],
                front_probabilities=prediction_data.get('front_probabilities', ''),
                back_probabilities=prediction_data.get('back_probabilities', '')
            )
            session.add(record)
            session.commit()
            return record.id
        except Exception as e:
            session.rollback()
            print(f"保存预测记录出错: {str(e)}")
            raise
        finally:
            session.close()
    
    def get_predictions(self, limit=100):
        """获取预测记录"""
        session = self.get_session()
        try:
            predictions = session.query(PredictionRecord).order_by(PredictionRecord.id.desc()).limit(limit).all()
            return predictions
        except Exception as e:
            session.rollback()
            print(f"查询预测记录出错: {str(e)}")
            return []
        finally:
            session.close()
    
    def update_prediction_result(self, prediction_id, actual_front, actual_back, front_match, back_match):
        """更新预测结果"""
        session = self.get_session()
        try:
            record = session.query(PredictionRecord).filter(PredictionRecord.id == prediction_id).first()
            if record:
                record.actual_front = actual_front
                record.actual_back = actual_back
                record.front_match_count = front_match
                record.back_match_count = back_match
                record.is_matched = 1 if (front_match + back_match) > 0 else 0
                
                if front_match == 5 and back_match == 2:
                    record.match_level = "一等奖"
                elif front_match == 5 and back_match == 1:
                    record.match_level = "二等奖"
                elif front_match == 5 and back_match == 0:
                    record.match_level = "三等奖"
                elif front_match == 4 and back_match == 2:
                    record.match_level = "四等奖"
                elif front_match == 4 and back_match == 1:
                    record.match_level = "五等奖"
                elif (front_match == 3 and back_match == 2) or (front_match == 4 and back_match == 0):
                    record.match_level = "六等奖"
                elif (front_match == 2 and back_match == 2) or (front_match == 3 and back_match == 1) or back_match == 2:
                    record.match_level = "七等奖"
                elif (front_match == 3 and back_match == 0) or (front_match == 2 and back_match == 1) or (front_match == 1 and back_match == 2) or back_match == 1:
                    record.match_level = "八等奖"
                else:
                    record.match_level = "未中奖"
                
                session.commit()
                return True
            return False
        except Exception as e:
            session.rollback()
            print(f"更新预测结果出错: {str(e)}")
            return False
        finally:
            session.close()

if __name__ == "__main__":
    db = Database()
    print("数据库初始化成功")
