import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
from datetime import datetime
import time

class DaLeTouScraper:
    def __init__(self):
        self.base_url = "http://datachart.500.com/dlt/history/newinc/history.php"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def scrape_historical_data(self, start_year=2023, end_year=2025, limit=500):
        print(f"正在抓取历史开奖数据（最多{limit}条）...")
        all_data = []
        
        try:
            params = {'limit': limit}
            response = requests.get(self.base_url, params=params, headers=self.headers, timeout=20)
            response.encoding = 'gb2312'
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                all_rows = soup.find_all('tr')
                
                for row in all_rows:
                    cols = row.find_all('td')
                    
                    if len(cols) >= 15:
                        try:
                            text_data = [col.text.strip() for col in cols]
                            
                            if text_data[0].isdigit() and len(text_data[0]) == 5:
                                period = text_data[0]
                                date_str = text_data[14] if len(text_data) > 14 else ''
                                
                                front_nums = []
                                back_nums = []
                                
                                for i in range(1, 6):
                                    if text_data[i].isdigit():
                                        front_nums.append(int(text_data[i]))
                                
                                for i in range(6, 8):
                                    if text_data[i].isdigit():
                                        back_nums.append(int(text_data[i]))
                                
                                if len(front_nums) == 5 and len(back_nums) == 2:
                                    all_data.append({
                                        'period': period,
                                        'date': date_str,
                                        'front_1': front_nums[0],
                                        'front_2': front_nums[1],
                                        'front_3': front_nums[2],
                                        'front_4': front_nums[3],
                                        'front_5': front_nums[4],
                                        'back_1': back_nums[0],
                                        'back_2': back_nums[1]
                                    })
                        except (ValueError, IndexError) as e:
                            continue
                
        except Exception as e:
            print(f"抓取数据时出错: {str(e)}")
        
        df = pd.DataFrame(all_data)
        print(f"成功获取 {len(df)} 条数据")
        if len(df) > 0:
            print(f"期号范围: {df['period'].iloc[0]} 到 {df['period'].iloc[-1]}")
        return df
    
    def save_to_csv(self, df, filename='lottery_data.csv'):
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"数据已保存到 {filename}")
        return filename

def main():
    scraper = DaLeTouScraper()
    df = scraper.scrape_historical_data(start_year=2023, end_year=2025)
    
    if not df.empty:
        scraper.save_to_csv(df)
        print(f"成功抓取 {len(df)} 条数据")
        print(df.head())
    else:
        print("未能获取到数据")

if __name__ == "__main__":
    main()
