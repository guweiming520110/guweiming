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
    
    front_numbers = Column(String(200))
    back_numbers = Column(String(100))
    
    front_probabilities = Column(String(1000))
    back_probabilities = Column(String(500))
    
    # 胆拖相关字段
    front_dan = Column(String(50))
    front_tuo = Column(String(200))
    back_dan = Column(String(20))
    back_tuo = Column(String(50))
    combination_count = Column(Integer)
    
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
                back_probabilities=prediction_data.get('back_probabilities', ''),
                front_dan=prediction_data.get('front_dan'),
                front_tuo=prediction_data.get('front_tuo'),
                back_dan=prediction_data.get('back_dan'),
                back_tuo=prediction_data.get('back_tuo'),
                combination_count=prediction_data.get('combination_count')
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
    
    def get_all_predictions_with_results(self):
        """获取所有预测记录及其对比结果"""
        session = self.get_session()
        try:
            predictions = session.query(PredictionRecord).order_by(PredictionRecord.id.desc()).all()
            results = []
            for pred in predictions:
                result = {
                    'id': pred.id,
                    'period': pred.target_period,
                    'prediction_date': pred.predict_time,
                    'play_type': pred.play_type,
                    'front_numbers': pred.front_numbers,
                    'back_numbers': pred.back_numbers,
                    'actual_front': pred.actual_front,
                    'actual_back': pred.actual_back,
                    'front_match_count': pred.front_match_count,
                    'back_match_count': pred.back_match_count,
                    'match_level': pred.match_level,
                    'is_matched': pred.is_matched,
                    'front_probabilities': pred.front_probabilities,
                    'back_probabilities': pred.back_probabilities
                }
                
                # 胆拖相关字段（如果存在）
                if hasattr(pred, 'front_dan'):
                    result['front_dan'] = pred.front_dan
                    result['front_tuo'] = pred.front_tuo
                    result['back_dan'] = pred.back_dan
                    result['back_tuo'] = pred.back_tuo
                    result['combination_count'] = pred.combination_count
                else:
                    result['front_dan'] = None
                    result['front_tuo'] = None
                    result['back_dan'] = None
                    result['back_tuo'] = None
                    result['combination_count'] = None
                
                results.append(result)
            return results
        finally:
            session.close()
    
    def get_prediction_statistics(self):
        """获取预测统计数据"""
        session = self.get_session()
        try:
            all_predictions = session.query(PredictionRecord).all()
            
            if not all_predictions:
                return {
                    'total_predictions': 0,
                    'verified_predictions': 0,
                    'by_play_type': {},
                    'by_match_level': {},
                    'dantuo_success_rate': {}
                }
            
            total = len(all_predictions)
            verified = len([p for p in all_predictions if p.actual_front is not None])
            
            # 按玩法统计
            by_play_type = {}
            for pred in all_predictions:
                play_type = pred.play_type or 'basic'
                if play_type not in by_play_type:
                    by_play_type[play_type] = {
                        'total': 0,
                        'verified': 0,
                        'matched': 0,
                        'avg_front_match': 0,
                        'avg_back_match': 0
                    }
                by_play_type[play_type]['total'] += 1
                if pred.actual_front is not None:
                    by_play_type[play_type]['verified'] += 1
                    if pred.is_matched:
                        by_play_type[play_type]['matched'] += 1
                    by_play_type[play_type]['avg_front_match'] += (pred.front_match_count or 0)
                    by_play_type[play_type]['avg_back_match'] += (pred.back_match_count or 0)
            
            # 计算平均值
            for play_type in by_play_type:
                verified_count = by_play_type[play_type]['verified']
                if verified_count > 0:
                    by_play_type[play_type]['avg_front_match'] /= verified_count
                    by_play_type[play_type]['avg_back_match'] /= verified_count
                    by_play_type[play_type]['success_rate'] = by_play_type[play_type]['matched'] / verified_count * 100
            
            # 按中奖等级统计
            by_match_level = {}
            for pred in all_predictions:
                if pred.match_level:
                    level = pred.match_level
                    by_match_level[level] = by_match_level.get(level, 0) + 1
            
            # 胆拖推荐成功率（检查胆码是否全部命中）
            dantuo_stats = {
                'total': 0,
                'front_dan_full_hit': 0,
                'back_dan_full_hit': 0,
                'both_dan_full_hit': 0
            }
            
            for pred in all_predictions:
                # 只处理胆拖推荐且有胆拖字段的记录
                if pred.play_type == 'dantuo' and pred.actual_front is not None and hasattr(pred, 'front_dan'):
                    dantuo_stats['total'] += 1
                    
                    if pred.front_dan and pred.actual_front:
                        try:
                            front_dan_list = list(map(int, pred.front_dan.split(',')))
                            actual_front_list = list(map(int, pred.actual_front.split(',')))
                            if set(front_dan_list).issubset(set(actual_front_list)):
                                dantuo_stats['front_dan_full_hit'] += 1
                        except:
                            pass
                    
                    if pred.back_dan and pred.actual_back:
                        try:
                            back_dan_list = list(map(int, pred.back_dan.split(',')))
                            actual_back_list = list(map(int, pred.actual_back.split(',')))
                            if set(back_dan_list).issubset(set(actual_back_list)):
                                dantuo_stats['back_dan_full_hit'] += 1
                        except:
                            pass
                    
                    # 两者都命中
                    if pred.front_dan and pred.back_dan and pred.actual_front and pred.actual_back:
                        try:
                            front_dan_list = list(map(int, pred.front_dan.split(',')))
                            actual_front_list = list(map(int, pred.actual_front.split(',')))
                            back_dan_list = list(map(int, pred.back_dan.split(',')))
                            actual_back_list = list(map(int, pred.actual_back.split(',')))
                            if (set(front_dan_list).issubset(set(actual_front_list)) and 
                                set(back_dan_list).issubset(set(actual_back_list))):
                                dantuo_stats['both_dan_full_hit'] += 1
                        except:
                            pass
            
            # 计算胆码命中率
            if dantuo_stats['total'] > 0:
                dantuo_stats['front_dan_hit_rate'] = dantuo_stats['front_dan_full_hit'] / dantuo_stats['total'] * 100
                dantuo_stats['back_dan_hit_rate'] = dantuo_stats['back_dan_full_hit'] / dantuo_stats['total'] * 100
                dantuo_stats['both_dan_hit_rate'] = dantuo_stats['both_dan_full_hit'] / dantuo_stats['total'] * 100
            else:
                dantuo_stats['front_dan_hit_rate'] = 0
                dantuo_stats['back_dan_hit_rate'] = 0
                dantuo_stats['both_dan_hit_rate'] = 0
            
            return {
                'total_predictions': total,
                'verified_predictions': verified,
                'by_play_type': by_play_type,
                'by_match_level': by_match_level,
                'dantuo_success_rate': dantuo_stats
            }
        finally:
            session.close()

if __name__ == "__main__":
    db = Database()
    print("数据库初始化成功")
