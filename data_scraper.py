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
    
    def scrape_historical_data(self, start_year=2023, end_year=2025):
        all_data = []
        
        for year in range(start_year, end_year + 1):
            print(f"正在抓取 {year} 年数据...")
            try:
                params = {'year': year}
                response = requests.get(self.base_url, params=params, headers=self.headers, timeout=10)
                response.encoding = 'gb2312'
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    all_rows = soup.find_all('tr')
                    
                    for row in all_rows:
                        cols = row.find_all('td')
                        
                        if len(cols) >= 8:
                            try:
                                text_data = [col.text.strip() for col in cols]
                                
                                if text_data[0].isdigit() and len(text_data[0]) == 5:
                                    period = text_data[0]
                                    
                                    front_nums = []
                                    back_nums = []
                                    
                                    for i in range(1, 6):
                                        if i < len(text_data) and text_data[i].isdigit():
                                            front_nums.append(int(text_data[i]))
                                    
                                    for i in range(6, 8):
                                        if i < len(text_data) and text_data[i].isdigit():
                                            back_nums.append(int(text_data[i]))
                                    
                                    if len(front_nums) == 5 and len(back_nums) == 2:
                                        all_data.append({
                                            'period': period,
                                            'date': f'{year}-??-??',
                                            'front_1': front_nums[0],
                                            'front_2': front_nums[1],
                                            'front_3': front_nums[2],
                                            'front_4': front_nums[3],
                                            'front_5': front_nums[4],
                                            'back_1': back_nums[0],
                                            'back_2': back_nums[1]
                                        })
                            except (ValueError, IndexError):
                                continue
                    
                    time.sleep(1)
                
            except Exception as e:
                print(f"抓取 {year} 年数据时出错: {str(e)}")
                continue
        
        df = pd.DataFrame(all_data)
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
