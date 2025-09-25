# -*- coding: utf-8 -*-
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict
import sys
import traceback

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd

# ===== 設定 =====
OUTPUT_DIR = Path("./data")          # 出力フォルダ
OUTPUT_FILE = OUTPUT_DIR / "juggler_202509_all.csv"
PAGE_LOAD_TIMEOUT = 20               # 画面ロード待機の上限秒
TABLE_WAIT_TIMEOUT = 15              # テーブルが出るまで待つ上限秒

# IDと日付リスト
ids_dates = [
    ('2614722', '9/16(火)'),
    ('2616726', '9/17(水)'),
    ('2618808', '9/18(木)'),
    ('2620752', '9/19(金)'),
    ('2623197', '9/20(土)')
]


# 機種名とURLパーツ
juggler_kishu_dict: Dict[str, str] = {
    "マイジャグラーV": "%E3%83%9E%E3%82%A4%E3%82%B8%E3%83%A3%E3%82%B0%E3%83%A9%E3%83%BCV",
    "アイムジャグラーEX": "S%E3%82%A2%E3%82%A4%E3%83%A0%E3%82%B8%E3%83%A3%E3%82%B0%E3%83%A9%E3%83%BC%EF%BC%A5%EF%BC%B8",
    "ゴーゴージャグラー3": "%E3%82%B4%E3%83%BC%E3%82%B4%E3%83%BC%E3%82%B8%E3%83%A3%E3%82%B0%E3%83%A9%E3%83%BC%EF%BC%93",
    "ファンキージャグラー": "%E3%83%95%E3%82%A1%E3%83%B3%E3%82%AD%E3%83%BC%E3%82%B8%E3%83%A3%E3%82%B0%E3%83%A9%E3%83%BC%EF%BC%92%EF%BC%AB%EF%BC%B4",
    "ジャグラーガールズ": "%E3%82%B8%E3%83%A3%E3%82%B0%E3%83%A9%E3%83%BC%E3%82%AC%E3%83%BC%E3%83%AB%E3%82%BASS",
    "ウルトラミラクルジャグラー": "%E3%82%A6%E3%83%AB%E3%83%88%E3%83%A9%E3%83%9F%E3%83%A9%E3%82%AF%E3%83%AB%E3%82%B8%E3%83%A3%E3%82%B0%E3%83%A9%E3%83%BC",
    "ミスタージャグラー": "%E3%83%9F%E3%82%B9%E3%82%BF%E3%83%BC%E3%82%B8%E3%83%A3%E3%82%B0%E3%83%A9%E3%83%BC"
}

def build_driver() -> webdriver.Chrome:
    """Chrome を1回だけ起動（Selenium Manager 使用。Chrome/Edgeが入っていればOK）"""
    options = Options()
    options.add_argument("--headless=new")        # 新ヘッドレス
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    # ここで service を渡さないと Selenium Manager が自動解決してくれる
    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
    return driver

def wait_tables(driver: webdriver.Chrome):
    """テーブルが描画されるまで待機"""
    WebDriverWait(driver, TABLE_WAIT_TIMEOUT).until(
        EC.presence_of_all_elements_located((By.TAG_NAME, "table"))
    )

def scrape_one(driver: webdriver.Chrome, page_id: str, date_txt: str, kishu: str, kishu_url: str) -> List[Dict]:
    url = f"https://min-repo.com/{page_id}/?kishu={kishu_url}"
    print(f"取得中: {date_txt} {kishu} -> {url}")
    driver.get(url)
    wait_tables(driver)

    soup = BeautifulSoup(driver.page_source, "html.parser")
    tables = soup.find_all("table")

    rows_out: List[Dict] = []
    for table in tables:
        rows = table.find_all("tr")
        if not rows:
            continue
        headers = [th.get_text(strip=True) for th in rows[0].find_all(["th", "td"])]
        if "台番" in headers or "台番号" in headers:
            for tr in rows[1:]:
                cols = [td.get_text(strip=True) for td in tr.find_all("td")]
                if cols and len(cols) == len(headers):
                    row = dict(zip(headers, cols))
                    row["機種"] = kishu
                    row["日付"] = date_txt
                    row["ID"] = page_id
                    rows_out.append(row)
    return rows_out

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_data: List[Dict] = []

    driver = build_driver()
    try:
        for page_id, date_txt in ids_dates:
            for kishu, kishu_url in juggler_kishu_dict.items():
                try:
                    part = scrape_one(driver, page_id, date_txt, kishu, kishu_url)
                    all_data.extend(part)
                except Exception as e:
                    print(f"[WARN] 失敗: {date_txt} {kishu} ({page_id}) -> {e}")
                    traceback.print_exc(file=sys.stdout)
    finally:
        driver.quit()

    if not all_data:
        print("[INFO] 収集ゼロ。サイト構造変更/ブロックの可能性があります。")
        return

    df = pd.DataFrame(all_data)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"【全日・全機種の台データを保存完了】-> {OUTPUT_FILE.resolve()}  行数: {len(df)}")

if __name__ == "__main__":
    main()
