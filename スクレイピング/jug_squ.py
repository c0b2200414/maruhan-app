# -*- coding: utf-8 -*-
"""
min-repo の「マルハン入間店」タグ配下を巡回して
  1) 一覧ページから (page_id, 日付) を収集
  2) 各 page_id × ジャグラー機種の台データを取得
  3) CSVに保存
に対応した統合スクリプト。

使い方(例):
  python combined_scraper.py ^
    --tag-url "https://min-repo.com/tag/%E3%83%9E%E3%83%AB%E3%83%8F%E3%83%B3%E5%85%A5%E9%96%93%E5%BA%97/" ^
    --max-pages 6 ^
    --month 9 ^
    --out "data/juggler_2025-09_all.csv"

依存: selenium, bs4, pandas, (Chrome/Edge がローカルにインストールされていること)
"""

from __future__ import annotations
import re
import sys
import traceback
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# ---------------- 設定（必要なら編集） ----------------

# 対象となるジャグラー機種と URL パラメータ（squreiping.py から統合）
JUGGLER_KISHU_DICT: Dict[str, str] = {
    "マイジャグラーV": "%%E3%%83%%9E%%E3%%82%%A4%%E3%%82%%B8%%E3%%83%%A3%%E3%%82%%B0%%E3%%83%%A9%%E3%%83%%BCV",
    "アイムジャグラーEX": "S%%E3%%82%%A2%%E3%%82%%A4%%E3%%83%%A0%%E3%%82%%B8%%E3%%83%%A3%%E3%%82%%B0%%E3%%83%%A9%%E3%%83%%BC%%EF%%BC%%A5%%EF%%BC%%B8",
    "ゴーゴージャグラー3": "%%E3%%82%%B4%%E3%%83%%BC%%E3%%82%%B4%%E3%%83%%BC%%E3%%82%%B8%%E3%%83%%A3%%E3%%82%%B0%%E3%%83%%A9%%E3%%83%%BC%%EF%%BC%%93",
    "ファンキージャグラー": "%%E3%%83%%95%%E3%%82%%A1%%E3%%83%%B3%%E3%%82%%AD%%E3%%83%%BC%%E3%%82%%B8%%E3%%83%%A3%%E3%%82%%B0%%E3%%83%%A9%%E3%%83%%BC%%EF%%BC%%92%%EF%%BC%%AB%%EF%%BC%%B4",
    "ジャグラーガールズ": "%%E3%%82%%B8%%E3%%83%%A3%%E3%%82%%B0%%E3%%83%%A9%%E3%%83%%BC%%E3%%82%%AC%%E3%%83%%BC%%E3%%83%%AB%%E3%%82%%BASS",
    "ウルトラミラクルジャグラー": "%%E3%%82%%A6%%E3%%83%%AB%%E3%%83%%88%%E3%%83%%A9%%E3%%83%%9F%%E3%%83%%A9%%E3%%82%%AF%%E3%%83%%AB%%E3%%82%%B8%%E3%%83%%A3%%E3%%82%%B0%%E3%%83%%A9%%E3%%83%%BC",
    "ミスタージャグラー": "%%E3%%83%%9F%%E3%%82%%B9%%E3%%82%%BF%%E3%%83%%BC%%E3%%82%%B8%%E3%%83%%A3%%E3%%82%%B0%%E3%%83%%A9%%E3%%83%%BC",
}

PAGE_LOAD_TIMEOUT = 20
TABLE_WAIT_TIMEOUT = 15

REQUIRED_OUTPUT_COLUMNS = ["日付", "ID", "機種", "台番", "機種名", "差枚", "G数", "BB", "RB", "合成"]  # 取得サイトの列に合わせて適宜調整

# ---------------- Selenium ヘルパ ----------------

def build_driver() -> webdriver.Chrome:
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(options=opts)
    driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
    return driver

def wait_tables(driver: webdriver.Chrome):
    WebDriverWait(driver, TABLE_WAIT_TIMEOUT).until(
        EC.presence_of_all_elements_located((By.TAG_NAME, "table"))
    )

# ---------------- 1) 一覧ページから (page_id, 日付) 収集 ----------------
# list.py の動きを統合：タグ一覧を走査し、リンクの href から 7～8桁の ID を抽出、aタグテキストから日付を拾う。:contentReference[oaicite:2]{index=2}

ID_RE = re.compile(r"/(\d{7,8})/")

def collect_ids_dates(tag_url: str, max_pages: int, month: int | None, driver: webdriver.Chrome) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for page in range(1, max_pages + 1):
        url = f"{tag_url}?page={page}"
        driver.get(url)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            m = ID_RE.search(href)
            if not m:
                continue
            page_id = m.group(1)
            date_txt = a.get_text(strip=True)  # "9/16(火)" など
            if month is not None:
                # "9/..." など月頭一致でフィルタ
                if not date_txt.startswith(f"{month}/"):
                    continue
            out.append((page_id, date_txt))
    # 重複除去（同じID/日付が複数回出る可能性に備える）
    seen = set()
    uniq = []
    for pid, dtx in out:
        key = (pid, dtx)
        if key not in seen:
            seen.add(key)
            uniq.append(key)
    return uniq

# ---------------- 2) page_id × 機種 → テーブル吸い上げ ----------------
# squreiping.py の scrape_one を統合し、各テーブルから行を抽出する。:contentReference[oaicite:3]{index=3}

def scrape_one(driver: webdriver.Chrome, page_id: str, date_txt: str, kishu: str, kishu_url: str) -> List[dict]:
    url = f"https://min-repo.com/{page_id}/?kishu={kishu_url}"
    print(f"[GET] {date_txt} {kishu} -> {url}")
    driver.get(url)
    wait_tables(driver)

    soup = BeautifulSoup(driver.page_source, "html.parser")
    tables = soup.find_all("table")
    rows_out: List[dict] = []

    for table in tables:
        trs = table.find_all("tr")
        if not trs:
            continue
        headers = [th.get_text(strip=True) for th in trs[0].find_all(["th", "td"])]

        # ざっくり「台番」列があるテーブルだけ採用（サイトの構造に合わせて調整してOK）
        if ("台番" in headers) or ("台番号" in headers):
            for tr in trs[1:]:
                tds = [td.get_text(strip=True) for td in tr.find_all("td")]
                if not tds or len(tds) != len(headers):
                    continue
                row = dict(zip(headers, tds))
                # 付加情報
                row["機種"] = kishu
                row["日付"] = date_txt
                row["ID"]   = page_id
                rows_out.append(row)

    return rows_out

# ---------------- 3) メイン ----------------

def main():
    ap = argparse.ArgumentParser(description="min-repo | マルハン入間店 | 一覧→台データ 収集統合スクリプト")
    ap.add_argument("--tag-url", required=True, help="タグ一覧のURL (例: https://min-repo.com/tag/.../)")
    ap.add_argument("--max-pages", type=int, default=6, help="一覧ページを何ページ分見るか")
    ap.add_argument("--month", type=int, default=None, help="特定の月だけ拾う (例: 9)")
    ap.add_argument("--out", default="data/juggler_all.csv", help="出力CSVパス")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    driver = build_driver()
    try:
        # 1) 一覧から (page_id, 日付) を収集
        ids_dates = collect_ids_dates(args.tag_url, args.max_pages, args.month, driver)
        if not ids_dates:
            print("[WARN] ID/日付が取得できませんでした。引数（URL/月/ページ数）やサイト構造を確認してください。")
            return

        print(f"[INFO] 収集 ID件数: {len(ids_dates)} 例: {ids_dates[:3]}")

        # 2) 各ID × 機種 で台データを吸い上げ
        all_rows: List[dict] = []
        for page_id, date_txt in ids_dates:
            for kishu, kishu_url in JUGGLER_KISHU_DICT.items():
                try:
                    part = scrape_one(driver, page_id, date_txt, kishu, kishu_url)
                    all_rows.extend(part)
                except Exception as e:
                    print(f"[WARN] 失敗: {date_txt} {kishu} ({page_id}) -> {e}")
                    traceback.print_exc(file=sys.stdout)

    finally:
        driver.quit()

    if not all_rows:
        print("[INFO] 収集ゼロ。サイト側の構造変更やブロックの可能性があります。")
        return

    df = pd.DataFrame(all_rows)

    # 列名の正規化（サイトの見出しに合わせて必要ならマッピング）
    # 例: "台番","機種","差枚","G数","BB","RB","合成" などを残す
    # 足りない列があっても、とりあえず全列を保存
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[DONE] 保存完了: {out_path.resolve()}  行数: {len(df)}")

if __name__ == "__main__":
    main()
