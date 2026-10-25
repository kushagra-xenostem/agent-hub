import io
import os
import time

import httpx
import pandas as pd
import streamlit as st
from PIL import Image

BACKEND = os.getenv("BACKEND_URL", f"http://127.0.0.1:{os.getenv('BACKEND_PORT', '8001')}")
RUNNING = "running"

st.set_page_config(page_title="Multi-Agent Hub", layout="wide")
ss = st.session_state


def api(method, path, **kw):
    return httpx.request(method, f"{BACKEND}{path}", timeout=kw.pop("timeout", 60), **kw)


def start(path, **kw):
    r = api("POST", path, **kw)
    if r.status_code != 200:
        st.error(r.json().get("detail", r.text))
        return None
    return r.json()["run_id"]


try:
    health = api("GET", "/health", params={"check_llm": True}).json()
except httpx.HTTPError:
    st.error(f"Backend not reachable at {BACKEND}")
    st.stop()

st.title("Multi-Agent Hub")
st.caption(f"LLM: {'connected' if health.get('llm') else 'unreachable'}  |  Browser automation: {health['browser']}  |  "
           "Agents run on the ARES runtime: validated tool calls, least privilege, recovery, injection screening")
if not health.get("llm"):
    st.warning("The LLM server is unreachable. Check LLM_BASE_URL and LLM_API_KEY in .env.")
tabs = st.tabs(["Research", "Database", "Browser", "Security (add-on)", "Evaluation (add-on)", "History"])

EVENT_LABEL = {"llm": "Model", "tool": "Tool", "tool_error": "Tool error", "error": "Error", "recovery": "Recovery",
               "security": "Security", "compaction": "Context", "wrapup": "Wrap-up", "sql": "SQL", "result": "Result",
               "answer": "Answer", "blocked": "Blocked", "sql_error": "SQL error", "step": "Step", "info": "Info",
               "eval_item": "Eval"}


def describe(e: dict) -> str:
    t = e.get("type")
    if t == "tool":
        args = e.get("args") or {}
        what = args.get("query") or args.get("url") or ""
        return f"{e['tool']}({what}) - {e.get('info', '')} [{e.get('ms', 0)} ms]"
    if t == "tool_error":
        return f"{e['tool']} failed: {e['error']}"
    if t == "llm":
        return f"model call {'ok' if e.get('ok') else 'failed'} [{e.get('ms', 0)} ms, {e.get('tokens') or '?'} tokens]"
    if t == "recovery":
        return f"{e.get('action')}: {e.get('detail', '')}"
    if t == "security":
        return f"{e.get('check')}: {e.get('detail')}"
    if t == "sql":
        return f"attempt {e.get('attempt')}: {e.get('explanation', '')}"
    if t == "result":
        return f"{e.get('rows')} rows, columns {e.get('columns')} [{e.get('ms')} ms]"
    if t == "step":
        return f"step {e.get('step')}: {e.get('action')} -> {e.get('result')}"
    if t == "eval_item":
        return f"{e['n']}/{e['total']} {'correct' if e['correct'] else 'WRONG'}: {e['question']}"
    return str(e.get("message") or e.get("text") or e.get("reason") or "")


def timeline(events: list[dict], height: int = 260) -> None:
    with st.container(height=height, border=True):
        for e in events[-80:]:
            ts = time.strftime("%H:%M:%S", time.localtime(e.get("ts", 0)))
            label = EVENT_LABEL.get(e.get("type"), e.get("type", ""))
            st.markdown(f"<span style='font-family:monospace;font-size:0.8rem;color:"
                        f"{'#b91c1c' if 'error' in e.get('type', '') or e.get('type') == 'blocked' else '#6b7280'}'>"
                        f"{ts} {label:<10}</span> {describe(e)}", unsafe_allow_html=True)


def watch(key: str, render_result, height: int = 260) -> None:
    rid = ss.get(key)
    if not rid:
        return

    @st.fragment(run_every=1.5)
    def _live():
        r = api("GET", f"/runs/{rid}").json()
        status = r["status"]
        c1, c2 = st.columns([4, 1])
        c1.markdown(f"**Run {rid}** - {status}")
        if status == RUNNING and r["kind"] == "browser" and c2.button("Stop", key=f"stop_{rid}"):
            api("POST", f"/runs/{rid}/stop")
        timeline(r["events"], height)
        if status != RUNNING:
            if r.get("error"):
                st.error(r["error"])
            if r.get("result"):
                render_result(r)

    _live()


# ======================= Research =======================
def render_research(r):
    res = r["result"]
    rep = res.get("report") or {}
    st.subheader(rep.get("title", "No report"))
    if rep:
        st.write(rep.get("summary", ""))
        if not rep.get("answer_found", True):
            st.info("The agent reports that its sources do not answer the question.")
        for f in rep.get("findings", []):
            st.markdown(f"- {f['claim']}  \n  <small>[{f['source_url']}]({f['source_url']})</small>", unsafe_allow_html=True)
    else:
        st.warning(f"No report: {res.get('reason')}")
    st.caption(f"{res.get('llm_calls')} model calls, {res.get('tool_calls')} tool calls, {res.get('duration_s')} s"
               + (f", web: {res['web']}" if res.get("web") else ""))
    if res.get("fetched_urls"):
        with st.expander("Pages read"):
            for u in res["fetched_urls"]:
                st.markdown(f"- {u}")


with tabs[0]:
    st.caption("Ask a question; the agent searches, reads sources and writes a report where every finding cites a page it read.")
    with st.form("research"):
        q = st.text_area("Research question", height=80, placeholder="e.g. What is the Sun-Earth L2 point and which telescopes use it?")
        c1, c2 = st.columns([1, 2])
        source = c1.radio("Sources", ["Live web (Wikipedia + pages)", "Offline sample corpus"])
        extra = c2.text_area("Extra source URLs (optional, one per line)", height=80)
        steps = st.slider("Maximum model calls", 4, 20, 10)
        go = st.form_submit_button("Start research", type="primary")
    if go and q.strip():
        ss.research_run = start("/research", json={"question": q, "source": "offline" if source.startswith("Offline") else "live",
                                                   "extra_urls": [u.strip() for u in extra.splitlines() if u.strip()],
                                                   "max_llm_calls": steps})
    watch("research_run", render_research)

# ======================= Database =======================
def render_db(r):
    res = r["result"]
    if res.get("sql"):
        st.code(res["sql"], language="sql")
        if res.get("repaired"):
            st.caption("The first query failed and the agent repaired it.")
    st.markdown(f"**Answer:** {res.get('answer', '')}")
    out = res.get("result")
    if out and out["rows"]:
        df = pd.DataFrame(out["rows"], columns=out["columns"])
        st.dataframe(df, hide_index=True, use_container_width=True)
        if len(df.columns) == 2 and pd.api.types.is_numeric_dtype(df.iloc[:, 1]) and 1 < len(df) <= 40:
            st.bar_chart(df.set_index(df.columns[0]))
        st.download_button("Download CSV", df.to_csv(index=False), file_name="query_result.csv")


with tabs[1]:
    dbs = api("GET", "/databases").json()
    c1, c2 = st.columns([2, 1])
    choice = c1.selectbox("Database", [d["name"] for d in dbs]) if dbs else None
    db_id = next((d["id"] for d in dbs if d["name"] == choice), None)
    with c2.expander("Upload a database"):
        up = st.file_uploader("One .db/.sqlite file, or CSV files (each becomes a table)",
                              type=["db", "sqlite", "sqlite3", "csv"], accept_multiple_files=True)
        if up and st.button("Upload"):
            r = api("POST", "/databases/upload", files=[("files", (f.name, f.getvalue())) for f in up])
            st.success(f"Added with tables {r.json()['tables']}") if r.status_code == 200 else st.error(r.json()["detail"])
            if r.status_code == 200:
                st.rerun()
    if db_id:
        with st.expander("Schema"):
            for t in api("GET", f"/databases/{db_id}/schema").json():
                st.markdown(f"**{t['table']}** ({t['rows']} rows): " + ", ".join(f"{c['name']} {c['type']}" for c in t["columns"]))
        with st.form("db"):
            question = st.text_input("Ask a question about the data",
                                     placeholder="e.g. Revenue by product category in 2025, highest first")
            ask = st.form_submit_button("Ask", type="primary")
        if ask and question.strip():
            ss.db_run = start("/databases/ask", json={"db_id": db_id, "question": question})
        watch("db_run", render_db, height=150)
        with st.expander("Run SQL yourself (read-only)"):
            sql = st.text_area("SQL", "SELECT category, count(*) AS products FROM products GROUP BY category")
            if st.button("Run SQL"):
                r = api("POST", "/databases/query", json={"db_id": db_id, "sql": sql})
                if r.status_code == 200:
                    st.dataframe(pd.DataFrame(r.json()["rows"], columns=r.json()["columns"]), hide_index=True)
                else:
                    st.error(r.json()["detail"])

# ======================= Browser =======================
def render_browser(r):
    res = r["result"]
    (st.success if res["status"] == "done" else st.warning)(f"{res['status']}: {res['answer']}")
    st.caption(f"{len(res['steps'])} steps in {res['duration_s']} s; {res.get('blocked_actions', 0)} action(s) blocked by guardrails")
    for s in res["steps"]:
        with st.expander(f"Step {s['step']}: {(s.get('action') or {}).get('type', 'error')} - {s.get('title') or s.get('url', '')}"):
            c1, c2 = st.columns([1, 1])
            c1.markdown(f"**Thought:** {s.get('thought', '')}  \n**Action:** `{s.get('action')}`  \n**Result:** {s.get('result', s.get('error'))}")
            if s.get("screenshot"):
                img = api("GET", f"/runs/{r['id']}/screenshots/{s['screenshot']}")
                if img.status_code == 200:
                    c2.image(Image.open(io.BytesIO(img.content)), use_container_width=True)


with tabs[2]:
    st.caption("The agent operates a real browser (your installed Edge or Chrome). It never enters passwords or payment "
               "details, and only visits public http(s) sites.")
    with st.form("browser"):
        task = st.text_area("Task", height=70, placeholder="e.g. Find the price of 'Sharp Objects' in the Mystery category")
        c1, c2, c3 = st.columns([2, 2, 1])
        url = c1.text_input("Start URL (optional)", "https://books.toscrape.com/")
        domains = c2.text_input("Stay on these domains (optional, comma separated)", "books.toscrape.com")
        mx = c3.number_input("Max steps", 1, 40, 12)
        visible = st.checkbox("Show the browser window while it works")
        go_b = st.form_submit_button("Start", type="primary")
    if go_b and task.strip():
        ss.browser_run = start("/browser", json={"task": task, "start_url": url or None, "max_steps": int(mx),
                                                 "allowed_domains": [d.strip() for d in domains.split(",") if d.strip()],
                                                 "headless": not visible})
    watch("browser_run", render_browser)

# ======================= Security =======================
with tabs[3]:
    st.caption("Prompt-injection and data-exfiltration attacks (OWASP LLM Top 10) against the research agent's defences.")
    attacks = api("GET", "/security/attacks").json()
    st.dataframe(pd.DataFrame(attacks), hide_index=True, use_container_width=True)
    c1, c2 = st.columns(2)
    if c1.button("Run guarantee matrix (offline, worst-case model)"):
        ss.sec_run = start("/security/matrix")
    pick = c2.selectbox("Live attack against the real model", [f"{a['id']}: {a['name']}" for a in attacks])
    if c2.button("Run live attack"):
        ss.sec_run = start("/security/live", json={"attack_id": pick.split(":")[0]})

    def render_sec(r):
        res = r["result"]
        if "rows" in res:
            df = pd.DataFrame(res["rows"])
            pivot = df.pivot_table(index="attack", columns="profile", values="succeeded", aggfunc="max")
            st.markdown("**Did the attack succeed?** (True = security failure). Columns remove one defence at a time.")
            st.dataframe(pivot, use_container_width=True)
        else:
            row = res["row"]
            ok = not row["succeeded"]
            (st.success if ok else st.error)(f"{row['name']}: attack {'blocked' if ok else 'SUCCEEDED'}; "
                                             f"model attempted it: {row['attempted']}; blocked actions: {row['blocked']}; "
                                             f"injections detected: {row['injections_detected']}")
    watch("sec_run", render_sec)

# ======================= Evaluation =======================
with tabs[4]:
    st.caption("Execution accuracy of the database agent on 15 labelled questions over the sample shop database.")
    if st.button("Run text-to-SQL evaluation"):
        ss.eval_run = start("/eval/sql")

    def render_eval(r):
        res = r["result"]
        m = st.columns(4)
        m[0].metric("Execution accuracy", f"{res['execution_accuracy']:.0%}")
        m[1].metric("Correct", f"{res['correct']}/{res['total']}")
        m[2].metric("Self-repaired queries", res["repaired"])
        m[3].metric("Avg seconds", res["avg_seconds"])
        st.dataframe(pd.DataFrame(res["items"]), hide_index=True, use_container_width=True)
    watch("eval_run", render_eval, height=200)

# ======================= History =======================
with tabs[5]:
    hist = api("GET", "/runs").json()
    if hist:
        df = pd.DataFrame(hist)
        df["created"] = pd.to_datetime(df["created"], unit="s").dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(df[["id", "kind", "title", "status", "created"]], hide_index=True, use_container_width=True)
        pick = st.selectbox("Open run", [h["id"] for h in hist])
        c1, c2 = st.columns([1, 6])
        if c1.button("Delete"):
            api("DELETE", f"/runs/{pick}")
            st.rerun()
        st.json(api("GET", f"/runs/{pick}").json().get("result") or {})
    else:
        st.caption("No runs yet.")
