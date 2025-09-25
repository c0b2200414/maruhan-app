import re
import time
import csv
from datetime import datetime
from typing import List, Tuple

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# ===============================
# ID & 日付リストの収集部分
# ===============================

ID_RE = re.compile(r"/(\d{7,8})/")
DATE_ONLY_RE = re.compile(r"^\s*\d{1,2}/\d{1,2}\([月火水木金土日]\)\s*$")

def text_looks_iruma(txt: str, strict_shop: bool) -> bool:
    txt = (txt or "").strip()
    if DATE_ONLY_RE.match(txt):  # "9/22(月)" の形式
        return True
    if strict_shop and "マルハン入間店" in txt:
        return True
    return False

def collect_ids_dates(tag_url: str, max_pages: int, month: int | None, driver, strict_shop_check: bool = False) -> List[Tuple[str,str]]:
    out = []
    for page in range(1, max_pages + 1):
        url = f"{tag_url}?page={page}"
        driver.get(url)
        time.sleep(1)
        soup = BeautifulSoup(driver.page_source, "html.parser")

        for a in soup.find_all("a", href=True):
            href = a["href"]
            m = ID_RE.search(href)
            if not m:
                continue
            page_id = m.group(1)
            date_txt = a.get_text(strip=True)

            # 月フィルタ
            if month is not None and not date_txt.startswith(f"{month}/"):
                continue

            # 判定
            if not text_looks_iruma(date_txt, strict_shop=strict_shop_check):
                continue

            out.append((page_id, date_txt))

    # 重複削除
    seen, uniq = set(), []
    for pid, dtx in out:
        if (pid, dtx) not in seen:
            seen.add((pid, dtx))
            uniq.append((pid, dtx))
    return uniq

# ===============================
# 各日付ページから台データを取る部分
# ===============================

def scrape_day_page(page_id: str, driver) -> pd.DataFrame:
    url = f"https://min-repo.com/{page_id}/"
    driver.get(url)
    time.sleep(1.5)

    soup = BeautifulSoup(driver.page_source, "html.parser")

    # 店名チェック
    if "マルハン入間店" not in soup.text:
        return pd.DataFrame()  # 他店舗は無視

    # 日付を取得
    date_el = soup.find("h2")
    date_txt = date_el.get_text(strip=True) if date_el else ""
    date_std = datetime.now().strftime("%Y-%m-%d")
    try:
        m = re.search(r"(\d{1,2})/(\d{1,2})", date_txt)
        if m:
            mm, dd = int(m.group(1)), int(m.group(2))
            date_std = f"2025-{mm:02d}-{dd:02d}"
    except:
        pass

    # テーブル解析
    tables = soup.find_all("table")
    if not tables:
        return pd.DataFrame()

    df = pd.read_html(str(tables[0]))[0]
    # 列名を整理
    df.columns = [c.strip() for c in df.columns]
    if "台番" not in df.columns:
        return pd.DataFrame()

    df["date"] = date_std
    return df

# ===============================
# メイン処理
# ===============================

def main():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    try:
        # 日付リスト取得（例: 9月、最大6ページ分）
        ids_dates = collect_ids_dates(
            tag_url="https://min-repo.com/tag/マルハン入間店/",
            max_pages=6,
            month=9,
            driver=driver,
            strict_shop_check=False
        )
        print("取得した日付リスト件数:", len(ids_dates))

        all_data = []
        for pid, dtx in ids_dates:
            print("取得中:", pid, dtx)
            df_day = scrape_day_page(pid, driver)
            if not df_day.empty:
                all_data.append(df_day)

        if all_data:
            df_all = pd.concat(all_data, ignore_index=True)
            df_all.to_csv("iruma_juggler_2025-09.csv", index=False, encoding="utf-8-sig")
            print("保存完了: iruma_juggler_2025-09.csv")
        else:
            print("データなし")

    finally:
        driver.quit()

if __name__ == "__main__":
    main()
