# -*- coding: utf-8 -*-
"""검색 품질 대시보드 (10-3).

평가셋 지표와 실사용 로그를 한 화면에서 본다. LLM을 호출하지 않는다.

    streamlit run src/dashboard.py
"""
import time

import pandas as pd
import streamlit as st

from src import tokenizer_ko
from src.build_sqlite import connect
from src.evaluate import MODES, evaluate, load

st.set_page_config(page_title="검색 품질 대시보드", layout="wide")
COLS = {"Hit@1": "hit@1", "Hit@5": "hit@5", "Recall@10": "recall@10",
        "MRR": "mrr", "p50(ms)": "p50_ms"}


@st.cache_resource
def get_conn():
    conn = connect()
    tokenizer_ko.warmup()             # Kiwi 사전 적재 (04-4)
    return conn


@st.cache_data(ttl=600)
def run_eval(kind, modes):
    items = [x for x in load() if not kind or x.get("kind") == kind]
    if not items:
        return pd.DataFrame(), 0
    res = evaluate(get_conn(), items, modes=modes)
    rows = [dict({"모드": m}, **{k: round(res[m][v], 3) for k, v in COLS.items()})
            for m in modes]
    return pd.DataFrame(rows), len(items)


@st.cache_data(ttl=30)
def load_log():
    try:
        return pd.DataFrame([dict(r) for r in get_conn().execute(
            "SELECT ts, query, n_results, n_tokens, latency_ms, coverage,"
            " invalid, refused FROM query_log ORDER BY id DESC LIMIT 200")])
    except Exception:
        return pd.DataFrame()


# ── 평가 지표 ────────────────────────────────────────────────
st.title("검색 품질 대시보드")

kind = st.selectbox("평가셋", ["manual", "auto", "(전체)"])
modes = st.multiselect("모드", MODES, default=MODES)
df, n = run_eval(None if kind == "(전체)" else kind, modes)

if df.empty:
    st.warning("평가셋이 비어 있습니다. python -m src.evaluate --make-auto 30")
else:
    st.caption(f"평가셋 {n}개 · 갱신 {time.strftime('%H:%M:%S')}")
    st.dataframe(df.style.highlight_max(subset=["Recall@10", "MRR"],
                                        color="#dff0d8"),
                 use_container_width=True, hide_index=True)
    best = df.loc[df["MRR"].idxmax()]
    c1, c2 = st.columns([1, 3])
    c1.metric("MRR 최고 모드", best["모드"], f"{best['MRR']:.3f}")
    c2.bar_chart(df.set_index("모드")[["Recall@10", "MRR"]], height=220)

# ── 평가셋을 바꾸면 결론이 뒤집힌다 ──────────────────────────
st.subheader("평가셋을 바꾸면 결론이 뒤집힙니다")
a, na = run_eval("auto", MODES)
m, nm = run_eval("manual", MODES)
if a.empty or m.empty:
    st.info("auto·manual 평가셋이 모두 있어야 비교할 수 있습니다.")
else:
    pick = lambda d, x: d.loc[d["모드"] == x, "MRR"].iloc[0]
    st.dataframe(pd.DataFrame({
        "모드": MODES,
        f"auto MRR ({na}개)": [pick(a, x) for x in MODES],
        f"manual MRR ({nm}개)": [pick(m, x) for x in MODES],
    }), hide_index=True, use_container_width=True)
    st.caption("자동 평가셋에서는 keyword가 1등, 사람이 쓴 질의에서는 hybrid가 1등입니다.")

# ── 실사용 로그 ──────────────────────────────────────────────
log = load_log()
st.subheader("최근 질의 로그")
if log.empty:
    st.info("아직 로그가 없습니다. 10-2의 봇을 먼저 돌려 보세요.")
else:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("질의 수", len(log))
    c2.metric("거절률", f"{log['refused'].mean() * 100:.0f}%")
    c3.metric("허위 인용", int((log["invalid"] > 0).sum()))
    c4.metric("평균 커버리지", f"{log['coverage'].dropna().mean():.2f}")
    st.dataframe(log, use_container_width=True, hide_index=True)
    st.line_chart(log.set_index("ts")[["latency_ms"]])

    st.subheader("결과가 부실한 질의")
    bad = log[(log["n_results"] < 3) | (log["refused"] == 1)]
    if bad.empty:
        st.success("부실 질의 없음")
    else:
        st.dataframe(bad[["ts", "query", "n_results", "refused"]],
                     use_container_width=True, hide_index=True)
