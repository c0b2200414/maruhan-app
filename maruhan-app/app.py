# app.py
# PC上の決め打ちファイルを読み込み → スマホは閲覧だけ
# 使い方は本文の「手順」を参照

import os
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np
import pandas as pd
import streamlit as st

# ====== ここだけあなたの環境に合わせて変更 ======
# 例: CSV or XLSX の絶対パス（どちらでもOK）
DATA_PATH = "data/maruhan_iruma_juggler_from_xlsx_utf8.csv"
# ================================================

JST = timezone(timedelta(hours=9))

# ---------- 読み込みユーティリティ ----------
def read_table_auto(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)

    # CSV想定：文字コード順に試す
    for enc in ("utf-8-sig", "utf-8", "cp932", "shift_jis"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path, encoding="latin1")

def coerce_numeric(df: pd.DataFrame, cols=("diff","spins","big","reg")) -> pd.DataFrame:
    d = df.copy()
    for c in cols:
        if c in d.columns:
            d[c] = (d[c].astype(str)
                        .str.replace(",", "", regex=False)
                        .str.replace(" ", "", regex=False)
                        .str.replace("\u3000", "", regex=False))
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d

# ---------- 設定5の合算・判定 ----------
@dataclass
class HeuristicConfig:
    min_spins_strong: int = 4000
    min_spins_negative_deny: int = 5000
    strong_diff_plus: int = 2000
    deny_diff_minus: int = -500
    fallback_gassan_thresh_setting5: float = 130.0
    recent_days: int = 30
    w_event_rate: float = 0.50
    w_weekday_rate: float = 0.25
    w_suffix_rate: float = 0.15
    w_recent_rate: float = 0.10
    laplace_alpha: float = 1.0

MODEL_THRESHOLDS: Dict[str, float] = {
    "マイジャグラーV": 126.8,
    "アイムジャグラーEX": 128.5,
    "ゴーゴージャグラー3": 123.7,
    "ファンキージャグラー": 133.2,
    "ジャグラーガールズ": 128.3,
    "ウルトラミラクルジャグラー": 130.8,
    "ミスタージャグラー": 124.4,
}

def rate_with_laplace(succ: int, tot: int, alpha: float=1.0) -> float:
    return (succ + alpha) / (tot + 2*alpha) if tot >= 0 else 0.0

def ensure_int(x):
    try: return int(x)
    except: return None

def label_high(spins, diff, big, reg, model: str, cfg: HeuristicConfig) -> int:
    spins = 0 if pd.isna(spins) else float(spins)
    diff  = 0 if pd.isna(diff)  else float(diff)
    big   = 0 if pd.isna(big)   else float(big)
    reg   = 0 if pd.isna(reg)   else float(reg)
    hits = big + reg
    gassan = (spins / hits) if hits > 0 else 9999.0
    th = MODEL_THRESHOLDS.get(str(model), cfg.fallback_gassan_thresh_setting5)

    if spins >= cfg.min_spins_negative_deny and diff <= cfg.deny_diff_minus: return 0
    if spins >= cfg.min_spins_strong and diff >= cfg.strong_diff_plus:       return 1
    if spins >= cfg.min_spins_strong and gassan <= th and diff >= 1000:      return 1
    if gassan <= (th - 3) and spins >= 3500:                                 return 1
    return 0

def normalize(s: pd.Series) -> pd.Series:
    x = s.astype(float).fillna(0.0)
    r = x.max() - x.min()
    return pd.Series(0.0, index=x.index) if r < 1e-9 else (x - x.min())/r

def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["date_dt"] = pd.to_datetime(d["date"], errors="coerce")
    d["date_dt"] = d["date_dt"].dt.tz_localize(JST, nonexistent="shift_forward", ambiguous="NaT")
    d["weekday"] = d["date_dt"].dt.weekday
    d["machine_no"] = d["machine_no"].astype(str)
    d["machine_no_int"] = d["machine_no"].apply(ensure_int)
    d["suffix"] = d["machine_no_int"].apply(lambda x: x % 10 if x is not None else None)
    hits = d["big"].fillna(0) + d["reg"].fillna(0)
    d["gassan"] = np.where(hits > 0, d["spins"] / hits.replace(0, np.nan), np.nan)
    return d

def make_bins(df: pd.DataFrame, cfg: HeuristicConfig) -> pd.DataFrame:
    d = df.copy()
    if "is_high_setting" in d.columns:
        d["is_high_setting_bin"] = pd.to_numeric(d["is_high_setting"], errors="coerce").fillna(0).clip(0,1)
    else:
        d["is_high_setting_bin"] = d.apply(
            lambda r: label_high(r["spins"], r["diff"], r["big"], r["reg"], r["model"], cfg),
            axis=1
        )
    return d

def build_event_scores(df, event_tag: Optional[str], cfg):
    d = df.copy()
    if event_tag is not None:
        d = d[d["event_tag"].fillna("") == event_tag]
    g = d.groupby("machine_no_int")["is_high_setting_bin"].agg(["sum","count"]).reset_index()
    g["event_rate"] = g.apply(lambda r: rate_with_laplace(int(r["sum"]), int(r["count"]), cfg.laplace_alpha), axis=1)
    return g[["machine_no_int","event_rate","count"]]

def build_weekday_scores(df, tgt_dt, cfg):
    d = df[df["weekday"] == tgt_dt.weekday()]
    g = d.groupby("machine_no_int")["is_high_setting_bin"].agg(["sum","count"]).reset_index()
    g["weekday_rate"] = g.apply(lambda r: rate_with_laplace(int(r["sum"]), int(r["count"]), cfg.laplace_alpha), axis=1)
    return g[["machine_no_int","weekday_rate","count"]]

def build_suffix_scores(df: pd.DataFrame, target_date, event_tag, cfg):
    d = df.copy()
    if event_tag is not None:
        d = d[d["event_tag"].fillna("") == event_tag]
    d = d[d["weekday"] == target_date.weekday()]
    d = d.dropna(subset=["suffix"])  # 末尾がNaNの行は除外

    # 名前付き集計で列名を固定
    g = (
        d.groupby("suffix")["is_high_setting_bin"]
         .agg(hs_sum="sum", n="count")
         .reset_index()
    )

    # ラプラス補正をベクトルで計算（apply不要）
    g["suffix_rate"] = (g["hs_sum"] + cfg.laplace_alpha) / (g["n"] + 2 * cfg.laplace_alpha)

    return g[["suffix", "suffix_rate"]]


def build_recent_scores(df, tgt_dt, cfg):
    start = (tgt_dt - timedelta(days=cfg.recent_days)).replace(hour=0, minute=0, second=0, microsecond=0)
    end   = (tgt_dt - timedelta(days=1)).replace(hour=23, minute=59, second=59, microsecond=0)
    d = df[(df["date_dt"]>=start) & (df["date_dt"]<=end)]
    g = d.groupby("machine_no_int")["is_high_setting_bin"].agg(["sum","count"]).reset_index()
    g["recent_rate"] = g.apply(lambda r: rate_with_laplace(int(r["sum"]), int(r["count"]), cfg.laplace_alpha), axis=1)
    return g[["machine_no_int","recent_rate","count"]]

def assemble_scores(df, tgt_dt, event_tag, topk, cfg):
    ev = build_event_scores(df, event_tag, cfg)
    wd = build_weekday_scores(df, tgt_dt, cfg)
    rc = build_recent_scores(df, tgt_dt, cfg)
    sf = build_suffix_scores(df, tgt_dt, event_tag, cfg)

    base = pd.DataFrame({"machine_no_int": sorted([x for x in df["machine_no_int"].dropna().unique() if x is not None])})
    out = base.merge(ev, on="machine_no_int", how="left") \
              .merge(wd, on="machine_no_int", how="left") \
              .merge(rc, on="machine_no_int", how="left")
    out["suffix"] = out["machine_no_int"] % 10
    out = out.merge(sf[["suffix","suffix_rate"]], on="suffix", how="left")

    out["event_rate_n"]   = normalize(out["event_rate"])
    out["weekday_rate_n"] = normalize(out["weekday_rate"])
    out["suffix_rate_n"]  = normalize(out["suffix_rate"])
    out["recent_rate_n"]  = normalize(out["recent_rate"])

    out["score"] = (
        cfg.w_event_rate   * out["event_rate_n"] +
        cfg.w_weekday_rate * out["weekday_rate_n"] +
        cfg.w_suffix_rate  * out["suffix_rate_n"] +
        cfg.w_recent_rate  * out["recent_rate_n"]
    )

    cols = ["machine_no_int","score","event_rate","weekday_rate","suffix_rate","recent_rate"]
    out = out.sort_values("score", ascending=False).reset_index(drop=True)
    return out[cols].head(topk)

# ---------- UI ----------
st.set_page_config(page_title="次の日の狙い台（スマホ閲覧用）", layout="wide")
st.title("🎰 次の日の狙い台（スマホ閲覧用）")
st.caption(f"データ: {DATA_PATH}")

# キャッシュして高速化、手動更新ボタンでリロード
@st.cache_data(ttl=300)
def load_data():
    df0 = read_table_auto(DATA_PATH)
    df0 = coerce_numeric(df0, cols=("diff","spins","big","reg"))
    for c in ["date","machine_no","model","event_tag"]:
        if c not in df0.columns:
            raise ValueError(f"CSV/Excelに必要な列がありません: {c}")
    df0["model"] = df0["model"].astype(str)
    df0["event_tag"] = df0["event_tag"].fillna("").astype(str)
    df0["machine_no"] = df0["machine_no"].astype(str)
    return df0

cols_top = st.columns(4)
with cols_top[0]:
    date_val = st.date_input("ターゲット日", value=datetime.now(JST).date())
with cols_top[1]:
    mode = st.radio("モード", ["日付×イベント","日付のみ"], horizontal=True)
with cols_top[2]:
    topk = st.number_input("TopK", 5, 100, 15, 1)
with cols_top[3]:
    if st.button("データを再読込（ファイル更新後に押す）"):
        load_data.clear()  # キャッシュクリア

df = load_data()

# イベント選択（PC側データに入ってるものから）
event_list = sorted(list(df.get("event_tag", pd.Series([""])).fillna("").astype(str).unique()))
ev = st.selectbox("イベント（指定しない場合は空欄）", [""] + [e for e in event_list if e != ""])
event_tag = ev if (ev and ev.strip() != "") else None

# 重み・直近日数
cols_w = st.columns(4)
w_event   = cols_w[0].number_input("重み：イベント", 0.0, 1.0, 0.50, 0.01)
w_weekday = cols_w[1].number_input("重み：曜日",   0.0, 1.0, 0.25, 0.01)
w_suffix  = cols_w[2].number_input("重み：末尾",   0.0, 1.0, 0.15, 0.01)
w_recent  = cols_w[3].number_input("重み：直近",   0.0, 1.0, 0.10, 0.01)
recent_days = st.slider("直近日数", 7, 120, 30, 1)

# 計算
cfg = HeuristicConfig(
    recent_days=int(recent_days),
    w_event_rate=float(w_event),
    w_weekday_rate=float(w_weekday),
    w_suffix_rate=float(w_suffix),
    w_recent_rate=float(w_recent),
)
df1 = add_derived(df)
df1 = make_bins(df1, cfg)
tgt_dt = datetime.combine(date_val, datetime.min.time()).replace(tzinfo=JST)

if mode == "日付×イベント":
    top_event = assemble_scores(df1, tgt_dt, event_tag, int(topk), cfg)
    top_date  = assemble_scores(df1, tgt_dt, None,     int(topk), cfg)
else:
    top_event = assemble_scores(df1, tgt_dt, None, int(topk), cfg)
    top_date  = assemble_scores(df1, tgt_dt, None, int(topk), cfg)

inter = set(top_event["machine_no_int"]).intersection(set(top_date["machine_no_int"]))
inter_df = pd.DataFrame({"machine_no_int": sorted(list(inter))})
base = df1[["machine_no_int","model"]].drop_duplicates()
top_event = top_event.merge(base, on="machine_no_int", how="left")
top_date  = top_date.merge(base, on="machine_no_int", how="left")
inter_df  = inter_df.merge(base, on="machine_no_int", how="left")

st.subheader("イベント基準 Top")
st.dataframe(top_event, width="stretch")
st.download_button("CSVダウンロード（イベント基準）",
                   data=top_event.to_csv(index=False).encode("utf-8-sig"),
                   file_name="candidates_event.csv")

st.subheader("日付基準 Top")
st.dataframe(top_date, width="stretch")
st.download_button("CSVダウンロード（日付基準）",
                   data=top_date.to_csv(index=False).encode("utf-8-sig"),
                   file_name="candidates_date.csv")

st.subheader("交差（最終候補）")
st.dataframe(inter_df, width="stretch")
st.download_button("CSVダウンロード（交差）",
                   data=inter_df.to_csv(index=False).encode("utf-8-sig"),
                   file_name="candidates_intersection.csv")

st.caption("※PC上のファイルを読み込んでいます。スマホは閲覧操作のみでOK（同一Wi-Fi推奨／外出先はngrok等で公開）。")
