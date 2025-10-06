import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from typing import List, Dict, Tuple
from datetime import datetime

class MultiSourceScraper:
    """多数据源采集器，从多个网站获取大乐透数据"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.sources = []
    
    def scrape_500_com(self, limit=500) -> Tuple[pd.DataFrame, str]:
        """从500彩票网采集数据"""
        try:
            print(f"正在从500彩票网采集数据...")
            url = "http://datachart.500.com/dlt/history/newinc/history.php"
            params = {'limit': limit}
            
            response = requests.get(url, params=params, headers=self.headers, timeout=15)
            response.encoding = 'gb2312'
            
            if response.status_code != 200:
                return pd.DataFrame(), "500彩票网(失败)"
            
            soup = BeautifulSoup(response.text, 'html.parser')
            all_rows = soup.find_all('tr')
            
            data = []
            for row in all_rows:
                cols = row.find_all('td')
                if len(cols) >= 15:
                    try:
                        text_data = [col.text.strip() for col in cols]
                        if text_data[0].isdigit() and len(text_data[0]) == 5:
                            period = text_data[0]
                            date_str = text_data[14] if len(text_data) > 14 else ''
                            
                            front_nums = [int(text_data[i]) for i in range(1, 6) if text_data[i].isdigit()]
                            back_nums = [int(text_data[i]) for i in range(6, 8) if text_data[i].isdigit()]
                            
                            if len(front_nums) == 5 and len(back_nums) == 2:
                                data.append({
                                    'period': period,
                                    'date': date_str,
                                    'front_1': front_nums[0],
                                    'front_2': front_nums[1],
                                    'front_3': front_nums[2],
                                    'front_4': front_nums[3],
                                    'front_5': front_nums[4],
                                    'back_1': back_nums[0],
                                    'back_2': back_nums[1],
                                    'source': '500.com'
                                })
                    except (ValueError, IndexError):
                        continue
            
            df = pd.DataFrame(data)
            print(f"  ✓ 500彩票网: 获取 {len(df)} 条数据")
            return df, "500彩票网"
            
        except Exception as e:
            print(f"  ✗ 500彩票网采集失败: {str(e)}")
            return pd.DataFrame(), "500彩票网(失败)"
    
    def scrape_sina(self, limit=100) -> Tuple[pd.DataFrame, str]:
        """从新浪彩票采集数据（备用源）"""
        try:
            print(f"正在从新浪彩票采集数据...")
            # 新浪彩票API接口
            url = "http://data.cp.sina.com.cn/lottery/kj/index"
            params = {
                'type': 'dlt',
                'limit': limit
            }
            
            response = requests.get(url, params=params, headers=self.headers, timeout=10)
            
            if response.status_code != 200:
                return pd.DataFrame(), "新浪彩票(失败)"
            
            # 尝试解析JSON数据
            try:
                json_data = response.json()
                data = []
                
                if 'data' in json_data:
                    for item in json_data['data']:
                        # 解析数据格式
                        period = item.get('qh', '')
                        date_str = item.get('kjrq', '')
                        numbers = item.get('kj', '').split(',')
                        
                        if len(numbers) >= 7:
                            front_nums = [int(numbers[i]) for i in range(5)]
                            back_nums = [int(numbers[i]) for i in range(5, 7)]
                            
                            data.append({
                                'period': period,
                                'date': date_str,
                                'front_1': front_nums[0],
                                'front_2': front_nums[1],
                                'front_3': front_nums[2],
                                'front_4': front_nums[3],
                                'front_5': front_nums[4],
                                'back_1': back_nums[0],
                                'back_2': back_nums[1],
                                'source': 'sina.com'
                            })
                
                df = pd.DataFrame(data)
                print(f"  ✓ 新浪彩票: 获取 {len(df)} 条数据")
                return df, "新浪彩票"
                
            except:
                # 如果不是JSON，尝试HTML解析
                return pd.DataFrame(), "新浪彩票(格式不支持)"
            
        except Exception as e:
            print(f"  ✗ 新浪彩票采集失败: {str(e)}")
            return pd.DataFrame(), "新浪彩票(失败)"
    
    def scrape_lottery_gov_cn(self, limit=100) -> Tuple[pd.DataFrame, str]:
        """从中国体育彩票官网采集数据"""
        try:
            print(f"正在从体彩官网采集数据...")
            # 体彩官网开奖接口
            url = "https://webapi.sporttery.cn/gateway/lottery/getHistoryPageListV1.qry"
            
            params = {
                'gameNo': '85',  # 大乐透游戏编号
                'provinceId': '0',
                'pageSize': str(limit),
                'isVerify': '1',
                'pageNo': '1'
            }
            
            response = requests.get(url, params=params, headers=self.headers, timeout=15)
            
            if response.status_code != 200:
                return pd.DataFrame(), "体彩官网(失败)"
            
            try:
                json_data = response.json()
                data = []
                
                if 'value' in json_data and 'list' in json_data['value']:
                    for item in json_data['value']['list']:
                        period = item.get('lotteryDrawNum', '')
                        date_str = item.get('lotteryDrawTime', '').split(' ')[0]
                        numbers = item.get('lotteryDrawResult', '').split()
                        
                        if len(numbers) >= 7:
                            front_nums = [int(numbers[i]) for i in range(5)]
                            back_nums = [int(numbers[i]) for i in range(5, 7)]
                            
                            data.append({
                                'period': period,
                                'date': date_str,
                                'front_1': front_nums[0],
                                'front_2': front_nums[1],
                                'front_3': front_nums[2],
                                'front_4': front_nums[3],
                                'front_5': front_nums[4],
                                'back_1': back_nums[0],
                                'back_2': back_nums[1],
                                'source': 'sporttery.cn'
                            })
                
                df = pd.DataFrame(data)
                print(f"  ✓ 体彩官网: 获取 {len(df)} 条数据")
                return df, "体彩官网"
                
            except:
                return pd.DataFrame(), "体彩官网(解析失败)"
                
        except Exception as e:
            print(f"  ✗ 体彩官网采集失败: {str(e)}")
            return pd.DataFrame(), "体彩官网(失败)"
    
    def merge_and_deduplicate(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        """合并多个数据源并去重，保留最完整的记录"""
        if not dataframes:
            return pd.DataFrame()
        
        # 过滤空DataFrame
        valid_dfs = [df for df in dataframes if not df.empty]
        
        if not valid_dfs:
            return pd.DataFrame()
        
        # 合并所有数据
        merged_df = pd.concat(valid_dfs, ignore_index=True)
        
        # 转换期号为整数以便正确排序
        merged_df['period_int'] = merged_df['period'].astype(int)
        
        # 计算数据完整性评分（分数越高越完整）
        def calculate_completeness_score(row):
            score = 0
            # 有日期加2分
            if row['date'] and row['date'] != '':
                score += 2
            # 所有前区号码都有效加5分
            if all(pd.notna(row[f'front_{i}']) and row[f'front_{i}'] > 0 for i in range(1, 6)):
                score += 5
            # 所有后区号码都有效加3分
            if all(pd.notna(row[f'back_{i}']) and row[f'back_{i}'] > 0 for i in range(1, 3)):
                score += 3
            return score
        
        merged_df['completeness'] = merged_df.apply(calculate_completeness_score, axis=1)
        
        # 按期号（整数）降序、完整性评分降序排序，保留每个期号最完整的记录
        merged_df = merged_df.sort_values(['period_int', 'completeness'], ascending=[False, False])
        merged_df = merged_df.drop_duplicates(subset=['period'], keep='first')
        
        # 删除辅助列
        merged_df = merged_df.drop(['period_int', 'completeness'], axis=1)
        
        # 最终按期号（转为整数）降序排序
        merged_df['period_int'] = merged_df['period'].astype(int)
        merged_df = merged_df.sort_values('period_int', ascending=False)
        merged_df = merged_df.drop('period_int', axis=1)
        
        return merged_df
    
    def scrape_latest_updates(self, latest_period=None) -> Dict:
        """只获取最新数据（增量更新）"""
        print(f"\n========== 增量更新采集开始 ==========")
        if latest_period:
            print(f"数据库最新期号: {latest_period}")
            print(f"只获取最近50条并过滤已存在数据")
        else:
            print(f"数据库为空，获取最近50条历史数据")
        
        all_dataframes = []
        successful_sources = []
        
        # 只采集最近50条（速度快）
        limit = 50
        
        # 1. 500彩票网
        df_500, source_500 = self.scrape_500_com(limit)
        if not df_500.empty:
            all_dataframes.append(df_500)
            successful_sources.append(source_500)
        
        time.sleep(0.5)
        
        # 2. 体彩官网（官方源，优先级高）
        df_gov, source_gov = self.scrape_lottery_gov_cn(limit)
        if not df_gov.empty:
            all_dataframes.append(df_gov)
            successful_sources.append(source_gov)
        
        # 合并并去重
        final_df = self.merge_and_deduplicate(all_dataframes)
        
        # 如果有最新期号，只保留更新的数据
        if latest_period and not final_df.empty:
            latest_period_int = int(latest_period)
            final_df['period_int'] = final_df['period'].astype(int)
            final_df = final_df[final_df['period_int'] > latest_period_int]
            final_df = final_df.drop('period_int', axis=1)
            print(f"过滤后新增数据: {len(final_df)} 条")
        
        # 移除source列
        if 'source' in final_df.columns:
            final_df = final_df.drop('source', axis=1)
        
        print(f"\n========== 采集完成 ==========")
        print(f"总计获取: {len(final_df)} 条新数据")
        print(f"数据源: {', '.join(successful_sources)}")
        
        return {
            'dataframe': final_df,
            'sources': successful_sources,
            'total_count': len(final_df)
        }
    
    def scrape_all_sources(self, limit=500) -> Dict:
        """从所有数据源采集数据（全量采集）"""
        print(f"\n========== 多数据源采集开始 ==========")
        print(f"目标数量: {limit}条")
        
        all_dataframes = []
        successful_sources = []
        
        # 1. 500彩票网（主要数据源）
        df_500, source_500 = self.scrape_500_com(limit)
        if not df_500.empty:
            all_dataframes.append(df_500)
            successful_sources.append(source_500)
        
        # 短暂延迟
        time.sleep(1)
        
        # 2. 新浪彩票（备用源）
        df_sina, source_sina = self.scrape_sina(min(100, limit))
        if not df_sina.empty:
            all_dataframes.append(df_sina)
            successful_sources.append(source_sina)
        
        time.sleep(1)
        
        # 3. 体彩官网（官方源）
        df_gov, source_gov = self.scrape_lottery_gov_cn(min(100, limit))
        if not df_gov.empty:
            all_dataframes.append(df_gov)
            successful_sources.append(source_gov)
        
        # 合并并去重
        final_df = self.merge_and_deduplicate(all_dataframes)
        
        # 移除source列（不需要存储到数据库）
        if 'source' in final_df.columns:
            final_df = final_df.drop('source', axis=1)
        
        print(f"\n========== 采集完成 ==========")
        print(f"总计获取: {len(final_df)} 条有效数据")
        print(f"数据源: {', '.join(successful_sources)}")
        
        return {
            'dataframe': final_df,
            'sources': successful_sources,
            'total_count': len(final_df)
        }

if __name__ == "__main__":
    scraper = MultiSourceScraper()
    result = scraper.scrape_all_sources(limit=500)
    
    df = result['dataframe']
    print(f"\n数据预览:")
    print(df.head())
