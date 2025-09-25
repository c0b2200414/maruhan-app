#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import pandas as pd
import numpy as np

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

JST = timezone(timedelta(hours=9))

# ================= ここが重要：ファイル読込・数値化 =================

def read_table_auto(path: str) -> pd.DataFrame:
    """
    CSV(UTF-8/UTF-8-SIG/cp932/shift_jis) と Excel(xlsx/xls) を自動判別して読む。
    拡張子が.csvでも実体がxlsxなら 'PK' ヘッダで検出して read_excel に切替。
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)

    try:
        with open(path, "rb") as f:
            sig = f.read(2)
        if sig == b"PK":  # XLSXのZipヘッダ
            return pd.read_excel(path)
    except Exception:
        pass

    for enc in ("utf-8-sig", "utf-8", "cp932", "shift_jis"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    # 最終手段（読めない文字は化けるが読み込み優先）
    return pd.read_csv(path, encoding="latin1")

def coerce_numeric(df: pd.DataFrame, cols=("diff","spins","big","reg")) -> pd.DataFrame:
    """数値列を確実に数値化。カンマ/空白/全角空白を除去してから to_numeric(coerce)"""
    d = df.copy()
    for c in cols:
        if c in d.columns:
            d[c] = (
                d[c]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace(" ", "", regex=False)
                .str.replace("\u3000", "", regex=False)  # 全角空白
            )
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d

# ================= Core logic =================

@dataclass
class HeuristicConfig:
    min_spins_strong: int = 4000
    min_spins_negative_deny: int = 5000
    strong_diff_plus: int = 2000
    deny_diff_minus: int = -500
    fallback_gassan_thresh_setting4: float = 135.0
    recent_days: int = 30
    w_event_rate: float = 0.50
    w_weekday_rate: float = 0.25
    w_suffix_rate: float = 0.15
    w_recent_rate: float = 0.10
    laplace_alpha: float = 1.0

MODEL_THRESHOLDS: Dict[str, float] = {
    "マイジャグラーV":       126.8,
    "アイムジャグラーEX":     128.5,
    "ゴーゴージャグラー3":    123.7,
    "ファンキージャグラー":   133.2,
    "ジャグラーガールズ":     128.3,
    "ウルトラミラクルジャグラー": 130.8,
    "ミスタージャグラー":     124.4,
}


def parse_date(s: str) -> datetime:
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(str(s), fmt).replace(tzinfo=JST)
        except Exception:
            pass
    return pd.to_datetime(s).to_pydatetime().replace(tzinfo=JST)

def rate_with_laplace(success: int, total: int, alpha: float = 1.0) -> float:
    return (success + alpha) / (total + 2 * alpha) if total >= 0 else 0.0

def ensure_int(x):
    try:
        return int(x)
    except Exception:
        return None

def label_high_setting_core(spins, diff, big, reg, model: str, model_thresholds: Dict[str, float], cfg: HeuristicConfig) -> int:
    spins = 0 if (spins is None or pd.isna(spins)) else float(spins)
    diff = 0 if (diff is None or pd.isna(diff)) else float(diff)
    big = 0 if (big is None or pd.isna(big)) else float(big)
    reg = 0 if (reg is None or pd.isna(reg)) else float(reg)
    total_hits = big + reg
    gassan = (spins / total_hits) if total_hits > 0 else 9999.0
    th = model_thresholds.get(str(model), cfg.fallback_gassan_thresh_setting4)
    if spins >= cfg.min_spins_negative_deny and diff <= cfg.deny_diff_minus:
        return 0
    if spins >= cfg.min_spins_strong:
        if diff >= cfg.strong_diff_plus:
            return 1
        if gassan <= th and diff >= 1000:
            return 1
    if gassan <= (th - 3) and spins >= 3500:
        return 1
    return 0

def normalize(series: pd.Series) -> pd.Series:
    s = series.astype(float).fillna(0.0)
    if s.max() - s.min() < 1e-9:
        return pd.Series(0.0, index=s.index)
    return (s - s.min()) / (s.max() - s.min())

def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["date_dt"] = d["date"].apply(parse_date)
    d["weekday"] = d["date_dt"].dt.weekday
    d["machine_no_int"] = d["machine_no"].apply(ensure_int)
    d["suffix"] = d["machine_no_int"].apply(lambda x: x % 10 if x is not None else None)
    hits = d["big"].fillna(0) + d["reg"].fillna(0)
    d["gassan"] = np.where(hits > 0, d["spins"] / hits.replace(0, np.nan), np.nan)
    return d

def compute_high_setting_label(df: pd.DataFrame, cfg: HeuristicConfig) -> pd.DataFrame:
    d = df.copy()
    if "is_high_setting" in d.columns:
        d["is_high_setting_bin"] = d["is_high_setting"].astype(float).fillna(0).clip(0, 1)
    else:
        d["is_high_setting_bin"] = d.apply(
            lambda r: label_high_setting_core(r["spins"], r["diff"], r["big"], r["reg"], r["model"], MODEL_THRESHOLDS, cfg),
            axis=1
        )
    return d

def build_event_scores(df: pd.DataFrame, event_tag: Optional[str], cfg: HeuristicConfig) -> pd.DataFrame:
    d = df.copy()
    if event_tag is not None:
        d = d[d["event_tag"].fillna("") == event_tag]
    grp = d.groupby("machine_no_int")["is_high_setting_bin"]
    counts = grp.agg(["sum","count"]).reset_index().rename(columns={"sum":"hs_sum","count":"n"})
    counts["event_rate"] = counts.apply(lambda r: rate_with_laplace(int(r["hs_sum"]), int(r["n"]), cfg.laplace_alpha), axis=1)
    return counts[["machine_no_int","event_rate","n"]]

def build_weekday_scores(df: pd.DataFrame, target_date: datetime, cfg: HeuristicConfig) -> pd.DataFrame:
    target_weekday = target_date.weekday()
    d = df[df["weekday"] == target_weekday]
    grp = d.groupby("machine_no_int")["is_high_setting_bin"]
    counts = grp.agg(["sum","count"]).reset_index().rename(columns={"sum":"hs_sum","count":"n"})
    counts["weekday_rate"] = counts.apply(lambda r: rate_with_laplace(int(r["hs_sum"]), int(r["n"]), cfg.laplace_alpha), axis=1)
    return counts[["machine_no_int","weekday_rate","n"]]

def build_suffix_scores(df: pd.DataFrame, target_date: datetime, event_tag: Optional[str], cfg: HeuristicConfig) -> pd.DataFrame:
    d = df.copy()
    if event_tag is not None:
        d = d[d["event_tag"].fillna("") == event_tag]
    target_weekday = target_date.weekday()
    d = d[d["weekday"] == target_weekday]
    grp = d.groupby("suffix")["is_high_setting_bin"]
    sfx = grp.agg(["sum","count"]).reset_index().rename(columns={"sum":"hs_sum","count":"n"})
    sfx["suffix_rate"] = sfx.apply(lambda r: rate_with_laplace(int(r["hs_sum"]), int(r["n"]), cfg.laplace_alpha), axis=1)
    return sfx

def build_recent_scores(df: pd.DataFrame, target_date: datetime, cfg: HeuristicConfig) -> pd.DataFrame:
    start = (target_date - timedelta(days=cfg.recent_days)).replace(hour=0, minute=0, second=0, microsecond=0)
    end = (target_date - timedelta(days=1)).replace(hour=23, minute=59, second=59, microsecond=0)
    d = df[(df["date_dt"] >= start) & (df["date_dt"] <= end)]
    grp = d.groupby("machine_no_int")["is_high_setting_bin"]
    counts = grp.agg(["sum","count"]).reset_index().rename(columns={"sum":"hs_sum","count":"n"})
    counts["recent_rate"] = counts.apply(lambda r: rate_with_laplace(int(r["hs_sum"]), int(r["n"]), cfg.laplace_alpha), axis=1)
    return counts[["machine_no_int","recent_rate","n"]]

def assemble_scores(df: pd.DataFrame, target_date: datetime, event_tag: Optional[str], topk: int, cfg: HeuristicConfig):
    ev = build_event_scores(df, event_tag, cfg)
    wd = build_weekday_scores(df, target_date, cfg)
    rc = build_recent_scores(df, target_date, cfg)
    sf = build_suffix_scores(df, target_date, event_tag, cfg)

    base = pd.DataFrame({"machine_no_int": sorted([x for x in df["machine_no_int"].dropna().unique() if x is not None])})
    out = base.merge(ev, on="machine_no_int", how="left") \
              .merge(wd, on="machine_no_int", how="left") \
              .merge(rc, on="machine_no_int", how="left")
    out["suffix"] = out["machine_no_int"] % 10
    out = out.merge(sf[["suffix","suffix_rate"]], on="suffix", how="left")

    out["event_rate_n"] = normalize(out["event_rate"])
    out["weekday_rate_n"] = normalize(out["weekday_rate"])
    out["suffix_rate_n"] = normalize(out["suffix_rate"])
    out["recent_rate_n"] = normalize(out["recent_rate"])

    out["score"] = (
        cfg.w_event_rate * out["event_rate_n"] +
        cfg.w_weekday_rate * out["weekday_rate_n"] +
        cfg.w_suffix_rate * out["suffix_rate_n"] +
        cfg.w_recent_rate * out["recent_rate_n"]
    )

    stat_cols = ["event_rate","weekday_rate","suffix_rate","recent_rate"]
    out = out.sort_values("score", ascending=False).reset_index(drop=True)
    top_event = out[["machine_no_int","score"] + stat_cols].copy().head(topk)
    return out, top_event

def run_selector(data_csv: str, mode: str, target_date_str: str, event_tag: Optional[str], topk: int,
                 recent_days: int, weights: Tuple[float,float,float,float]) -> Dict[str, pd.DataFrame]:
    df = read_table_auto(data_csv)

    # 必須列チェック & 足りなければ落とす
    required = ["date","machine_no","model","diff","spins","big","reg","event_tag"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"CSVに必要な列がありません: {c}")

    # 数値列を強制数値化（ここが今回のエラー対策の肝）
    df = coerce_numeric(df, cols=("diff","spins","big","reg"))

    # 型の安全化
    df["model"] = df["model"].astype(str)
    df["event_tag"] = df["event_tag"].fillna("").astype(str)
    df["machine_no"] = df["machine_no"].astype(str)

    df = add_derived_columns(df)
    cfg = HeuristicConfig(recent_days=recent_days,
                          w_event_rate=weights[0],
                          w_weekday_rate=weights[1],
                          w_suffix_rate=weights[2],
                          w_recent_rate=weights[3])
    df = compute_high_setting_label(df, cfg)

    tgt = datetime.strptime(target_date_str, "%Y-%m-%d").replace(tzinfo=JST)

    if mode == "日付×イベント":
        _, top_event = assemble_scores(df, tgt, event_tag, topk, cfg)
        _, top_date  = assemble_scores(df, tgt, None, topk, cfg)
    elif mode == "日付のみ":
        _, top_event = assemble_scores(df, tgt, None, topk, cfg)
        _, top_date  = assemble_scores(df, tgt, None, topk, cfg)
    else:
        raise ValueError("未知のモード")

    ev_set = set(top_event["machine_no_int"].tolist())
    dt_set = set(top_date["machine_no_int"].tolist())
    inter = sorted(list(ev_set & dt_set))

    base_info = df[["machine_no_int","model"]].drop_duplicates()
    top_event_out = top_event.merge(base_info, on="machine_no_int", how="left")
    top_date_out  = top_date.merge(base_info, on="machine_no_int", how="left")
    inter_out = pd.DataFrame({"machine_no_int": inter}).merge(top_event_out, on="machine_no_int", how="left", suffixes=("","_ev")) \
                                                     .merge(top_date_out, on="machine_no_int", how="left", suffixes=("","_dt"))
    return {
        "event": top_event_out,
        "date": top_date_out,
        "intersection": inter_out
    }

# ================= Tkinter GUI =================

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Maruhan Iruma Juggler — 次の日の狙い台セレクタ")
        self.geometry("840x640")

        self.data_csv_var = tk.StringVar(value="")
        self.mode_var = tk.StringVar(value="日付×イベント")
        self.date_var = tk.StringVar(value=datetime.now(JST).strftime("%Y-%m-%d"))
        self.event_var = tk.StringVar(value="（指定しない）")
        self.topk_var = tk.IntVar(value=15)
        self.recent_days_var = tk.IntVar(value=30)

        self.w_event = tk.DoubleVar(value=0.50)
        self.w_weekday = tk.DoubleVar(value=0.25)
        self.w_suffix = tk.DoubleVar(value=0.15)
        self.w_recent = tk.DoubleVar(value=0.10)

        self.events_list = ["（指定しない）"]
        self.dates_list = []

        self.create_widgets()

    def create_widgets(self):
        frm = ttk.Frame(self, padding=10)
        frm.pack(fill="both", expand=True)

        row = ttk.Frame(frm)
        row.pack(fill="x", pady=4)
        ttk.Label(row, text="データCSV/Excel:").pack(side="left")
        ttk.Entry(row, textvariable=self.data_csv_var, width=70).pack(side="left", padx=6)
        ttk.Button(row, text="参照...", command=self.browse_csv).pack(side="left")

        row = ttk.Frame(frm); row.pack(fill="x", pady=4)
        ttk.Label(row, text="モード:").pack(side="left")
        ttk.Combobox(row, textvariable=self.mode_var, values=["日付×イベント","日付のみ"], width=15, state="readonly").pack(side="left", padx=6)
        ttk.Label(row, text="イベント:").pack(side="left", padx=(12,0))
        self.event_cb = ttk.Combobox(row, textvariable=self.event_var, values=self.events_list, width=30, state="readonly")
        self.event_cb.pack(side="left", padx=6)

        row = ttk.Frame(frm); row.pack(fill="x", pady=4)
        ttk.Label(row, text="ターゲット日 (YYYY-MM-DD):").pack(side="left")
        ttk.Combobox(row, textvariable=self.date_var, values=self.dates_list, width=20).pack(side="left", padx=6)
        ttk.Label(row, text="TopK:").pack(side="left", padx=(12,0))
        ttk.Spinbox(row, from_=5, to=100, textvariable=self.topk_var, width=6).pack(side="left", padx=6)
        ttk.Label(row, text="直近日数:").pack(side="left", padx=(12,0))
        ttk.Spinbox(row, from_=7, to=120, textvariable=self.recent_days_var, width=6).pack(side="left", padx=6)

        row = ttk.LabelFrame(frm, text="重み（0〜1, 合計は自動正規化しません）")
        row.pack(fill="x", pady=6)
        ttk.Label(row, text="イベント").grid(row=0, column=0, padx=6, pady=4, sticky="e")
        ttk.Entry(row, textvariable=self.w_event, width=6).grid(row=0, column=1, padx=6)
        ttk.Label(row, text="曜日").grid(row=0, column=2, padx=6, sticky="e")
        ttk.Entry(row, textvariable=self.w_weekday, width=6).grid(row=0, column=3, padx=6)
        ttk.Label(row, text="末尾").grid(row=0, column=4, padx=6, sticky="e")
        ttk.Entry(row, textvariable=self.w_suffix, width=6).grid(row=0, column=5, padx=6)
        ttk.Label(row, text="直近").grid(row=0, column=6, padx=6, sticky="e")
        ttk.Entry(row, textvariable=self.w_recent, width=6).grid(row=0, column=7, padx=6)

        row = ttk.Frame(frm); row.pack(fill="x", pady=8)
        ttk.Button(row, text="CSV/Excelを読み込んで候補を計算", command=self.run_now).pack(side="left")
        ttk.Button(row, text="結果を保存（3つのCSV）", command=self.save_results).pack(side="left", padx=10)

        self.nb = ttk.Notebook(frm)
        self.nb.pack(fill="both", expand=True, pady=6)

        self.txt_event = tk.Text(self.nb, wrap="none")
        self.txt_date = tk.Text(self.nb, wrap="none")
        self.txt_inter = tk.Text(self.nb, wrap="none")

        self.nb.add(self.txt_event, text="イベント基準 Top")
        self.nb.add(self.txt_date, text="日付基準 Top")
        self.nb.add(self.txt_inter, text="交差（最終候補）")

        self.status = tk.StringVar(value="CSV/Excelを選択してください")
        ttk.Label(frm, textvariable=self.status).pack(fill="x", pady=4)

        default_path = os.path.join(os.path.dirname(__file__), "maruhan_iruma_juggler_from_xlsx.csv")
        if os.path.exists(default_path):
            self.data_csv_var.set(default_path)
            self.refresh_meta()

    def browse_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV/Excel","*.csv;*.xlsx;*.xls"),("All Files","*.*")])
        if not path:
            return
        self.data_csv_var.set(path)
        self.refresh_meta()

    def refresh_meta(self):
        try:
            df = read_table_auto(self.data_csv_var.get())
            # 数値列を強制数値化（ここ大事）
            df = coerce_numeric(df, cols=("diff","spins","big","reg"))

            # イベント一覧と日付候補を更新
            events = sorted(list(df.get("event_tag", pd.Series([""])).fillna("").astype(str).unique()))
            self.events_list = ["（指定しない）"] + [e for e in events if e != ""]
            self.event_cb["values"] = self.events_list
            self.event_cb.set("（指定しない）")

            dates = sorted(list(set(pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d").dropna().tolist())))
            self.date_var.set(datetime.now(JST).strftime("%Y-%m-%d"))
            self.nb.tab(0, text="イベント基準 Top")
            self.nb.tab(1, text="日付基準 Top")
            self.nb.tab(2, text="交差（最終候補）")
            self.status.set(f"読み込み完了。イベント候補 {len(self.events_list)-1} 件 / 日付 {len(dates)} 件")
        except Exception as e:
            messagebox.showerror("読み込みエラー", str(e))

    def run_now(self):
        try:
            data_csv = self.data_csv_var.get()
            mode = self.mode_var.get()
            date_str = self.date_var.get().strip()
            event = self.event_var.get()
            if event == "（指定しない）":
                event = None
            topk = int(self.topk_var.get())
            recent_days = int(self.recent_days_var.get())
            weights = (float(self.w_event.get()), float(self.w_weekday.get()), float(self.w_suffix.get()), float(self.w_recent.get()))

            res = run_selector(data_csv, mode, date_str, event, topk, recent_days, weights)
            self.show_df(self.txt_event, res["event"])
            self.show_df(self.txt_date, res["date"])
            self.show_df(self.txt_inter, res["intersection"])
            self._last_result = res
            self.status.set("計算完了")
        except Exception as e:
            messagebox.showerror("計算エラー", str(e))

    def save_results(self):
        try:
            if not hasattr(self, "_last_result"):
                messagebox.showinfo("情報", "まず『CSV/Excelを読み込んで候補を計算』を実行してください。")
                return
            dirpath = filedialog.askdirectory()
            if not dirpath:
                return
            res = self._last_result
            res["event"].to_csv(os.path.join(dirpath, "candidates_event.csv"), index=False, encoding="utf-8-sig")
            res["date"].to_csv(os.path.join(dirpath, "candidates_date.csv"), index=False, encoding="utf-8-sig")
            res["intersection"].to_csv(os.path.join(dirpath, "candidates_intersection.csv"), index=False, encoding="utf-8-sig")
            messagebox.showinfo("保存", "3つのCSVを保存しました。")
        except Exception as e:
            messagebox.showerror("保存エラー", str(e))

    def show_df(self, textbox: tk.Text, df: pd.DataFrame):
        textbox.delete("1.0", tk.END)
        cols = [c for c in ["machine_no_int","model","score","event_rate","weekday_rate","suffix_rate","recent_rate"] if c in df.columns] + \
               [c for c in df.columns if c not in ["machine_no_int","model","score","event_rate","weekday_rate","suffix_rate","recent_rate"]]
        try:
            txt = df[cols].to_string(index=False)
        except Exception:
            txt = df.to_string(index=False)
        textbox.insert(tk.END, txt)

if __name__ == "__main__":
    app = App()
    app.mainloop()
