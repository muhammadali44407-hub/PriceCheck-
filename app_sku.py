import os
import io
import re
import urllib.parse
import asyncio
import aiohttp
import pandas as pd
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="SKU → ASIN Exporter (eBay)", page_icon="📦", layout="wide")

# =========================
# Secrets
# =========================
CRAWLBASE_TOKEN = st.secrets.get("CRAWLBASE_TOKEN", "")
APP_PASSWORD = st.secrets.get("APP_PASSWORD", "")

if not CRAWLBASE_TOKEN:
    st.error("⚠️ Missing CRAWLBASE_TOKEN in Streamlit secrets.")
    st.stop()
if not APP_PASSWORD:
    st.error("⚠️ Missing APP_PASSWORD in Streamlit secrets.")
    st.stop()

# =========================
# Auth (simple password gate)
# =========================
if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False

def do_login(pw: str):
    if pw and pw == APP_PASSWORD:
        st.session_state.auth_ok = True
    else:
        st.error("Incorrect password.")

def do_logout():
    st.session_state.auth_ok = False

if not st.session_state.auth_ok:
    st.title("🔒 SKU → ASIN Exporter (eBay)")
    st.caption("Enter password to continue.")
    pw = st.text_input("Password", type="password")
    if st.button("Login"):
        do_login(pw)
    st.stop()

# =========================
# UI
# =========================
st.title("SKU → ASIN Exporter (eBay)")
st.caption("Reads ASINs from **Custom label (SKU)** by stripping the first 3 chars, fetches Amazon data, merges with your sheet, and compares prices.")
st.button("Logout", on_click=do_logout)

left, right = st.columns([3, 2])

with left:
    c1, c2 = st.columns(2)
    with c1:
        domain = st.text_input("Amazon domain", value="www.amazon.com.au")
    with c2:
        max_concurrency = st.slider("Max concurrent requests", min_value=1, max_value=20, value=20)

    upl = st.file_uploader(
        "Upload eBay export (.csv or .xlsx). Must contain **Custom label (SKU)** and **Current price**",
        type=["csv", "xlsx"]
    )

with right:
    st.subheader("Live log")
    log_box = st.container()
    progress_bar = st.progress(0, text="Idle")
    counter_txt = st.empty()

# =========================
# Helpers
# =========================
def sanitize_text(s: str) -> str:
    """Replace the standalone word 'amazon' (any case) with 'ams'."""
    if not isinstance(s, str):
        return s
    return re.sub(r'\bamazon\b', 'ams', s, flags=re.IGNORECASE)

def derive_asin_from_sku(sku: str) -> str:
    """Remove the first 3 characters from SKU to get ASIN candidate."""
    if not isinstance(sku, str):
        return ""
    base = sku.strip()
    if len(base) <= 3:
        return ""
    return base[3:].strip()

ASIN_RE = re.compile(r"^[A-Za-z0-9]{10}$")
def valid_asin(x: str) -> bool:
    return isinstance(x, str) and bool(ASIN_RE.match(x))

def parse_money(val):
    """
    Parse a price that may include currency symbols or commas into a float.
    Returns None if it can't parse.
    """
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val)
    # Remove everything except digits, dot, comma, minus
    s = re.sub(r"[^0-9\.,-]", "", s)
    if s == "":
        return None
    # If both comma and dot present, assume comma is thousands sep → remove commas
    if "," in s and "." in s:
        s = s.replace(",", "")
    else:
        # If only comma present, treat comma as decimal separator
        if "," in s and "." not in s:
            s = s.replace(",", ".")
    try:
        return float(s)
    except:
        return None

def render_log_line(entry: dict) -> str:
    asin = entry.get("_asin", "")
    msg = entry.get("_msg", "")
    ok = entry.get("_ok", False)
    title = entry.get("TITLE") or ""
    stock = entry.get("inStock")
    prime = entry.get("isPrime")
    emoji = "✅" if ok else "❌"
    title_snip = title[:60] + ("…" if title and len(title) > 60 else "")
    stock_txt = "inStock" if stock is True else "outOfStock" if stock is False else "stock?"
    prime_txt = "Prime" if prime is True else "NoPrime" if prime is False else "prime?"
    return f"{emoji} {asin} • {title_snip} • {stock_txt} • {prime_txt} • {msg}"

def update_progress(done: int, total: int):
    pct = int((done / total) * 100) if total else 0
    progress_bar.progress(pct, text=f"Processing {done}/{total}…")
    counter_txt.write(f"Processed {done} of {total}")

# =========================
# Crawlbase fetch
# =========================
async def fetch_asin(session: aiohttp.ClientSession, token: str, domain: str, asin: str, sem: asyncio.Semaphore):
    """Fetch single ASIN from Crawlbase (amazon-product-details)."""
    amazon_url = f"https://{domain}/dp/{asin}"
    encoded_url = urllib.parse.quote(amazon_url, safe="")
    api_url = f"https://api.crawlbase.com/?token={token}&url={encoded_url}&scraper=amazon-product-details"

    async with sem:
        try:
            async with session.get(api_url, timeout=60) as resp:
                try:
                    data = await resp.json(content_type=None)
                except Exception as e:
                    return {"_ok": False, "_msg": f"Parse error: {e}", "_asin": asin}

                if isinstance(data, dict) and 'error' in data:
                    return {"_ok": False, "_msg": f"API error: {data.get('error')}", "_asin": asin}

                body = (data or {}).get('body', {}) if isinstance(data, dict) else {}
                features = body.get('features', []) or []
                description = body.get('description', '') or ''
                images = body.get('highResolutionImages', []) or []
                html_features = '<br>'.join(features) if features else ''
                html_combined = f"{html_features}<br><br>{description}" if html_features else description
                images_cleaned = ', '.join(images) if isinstance(images, list) else ''

                return {
                    "_ok": True,
                    "_msg": "OK",
                    "_asin": asin,
                    "TITLE": sanitize_text(body.get('name')),
                    "body_html": sanitize_text(html_combined),
                    "price": body.get('rawPrice'),
                    "highResolutionImages": images_cleaned,
                    "Brand": sanitize_text(body.get('brand')),
                    "isPrime": body.get('isPrime'),
                    "inStock": body.get('inStock'),
                    "stockDetail": body.get('stockDetail')
                }
        except Exception as e:
            return {"_ok": False, "_msg": f"Exception: {e}", "_asin": asin}

async def process_batch(token: str, domain: str, asins: list, max_concurrency: int, progress_cb=None, log_cb=None):
    sem = asyncio.Semaphore(max_concurrency)
    results, total, done = [], len(asins), 0
    async with aiohttp.ClientSession() as session:
        tasks = []
        for a in asins:
            if valid_asin(a):
                tasks.append(fetch_asin(session, token, domain, a, sem))
            else:
                # mark invalid upfront
                results.append({
                    "_ok": False, "_msg": "invalid asin", "_asin": a,
                    "TITLE": "not found", "body_html": "not found", "price": "not found",
                    "highResolutionImages": "not found", "Brand": "not found",
                    "isPrime": "not found", "inStock": "not found", "stockDetail": "not found"
                })
        for coro in asyncio.as_completed(tasks):
            res = await coro
            results.append(res)
            done += 1
            if log_cb: log_cb(res)
            if progress_cb: progress_cb(done, total)
    return results

# =========================
# Main
# =========================
SHEET_PRICE_COL = "Current price"  # <— change this if your price column name is different

if upl is not None:
    # Read CSV/XLSX
    try:
        if upl.name.lower().endswith(".csv"):
            df_in = pd.read_csv(upl)
        else:
            df_in = pd.read_excel(upl)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

    required_cols = ["Custom label (SKU)", SHEET_PRICE_COL]
    for col in required_cols:
        if col not in df_in.columns:
            st.error(f"Input file must contain a column named '{col}'.")
            st.stop()

    # Derive ASIN from SKU (remove first 3 chars)
    df_in = df_in.copy()
    df_in["ASIN"] = df_in["Custom label (SKU)"].apply(derive_asin_from_sku)

    # Prepare ASIN list
    asins = df_in["ASIN"].fillna("").astype(str).tolist()
    st.success(f"Loaded {len(asins)} rows. Derived ASINs from Custom label (SKU).")

    if st.button("Run"):
        log_holder = log_box.empty()
        log_lines = []

        def log_cb(res: dict):
            line = render_log_line(res)
            log_lines.append(line)
            log_holder.write("\n".join(log_lines[-200:]))

        with st.spinner("Fetching Amazon details…"):
            results = asyncio.run(
                process_batch(
                    CRAWLBASE_TOKEN,
                    domain,
                    asins,
                    max_concurrency,
                    progress_cb=update_progress,
                    log_cb=log_cb
                )
            )

        # Amazon results -> DataFrame
        df_amz = pd.DataFrame(results)
        amz_cols = ["TITLE","body_html","price","highResolutionImages","Brand","isPrime","inStock","stockDetail"]
        for c in amz_cols:
            if c not in df_amz.columns:
                df_amz[c] = "not found"
        if "_asin" not in df_amz.columns:
            df_amz["_asin"] = ""

        # Merge onto original rows (left join to keep all original rows)
        df_merge = df_in.merge(
            df_amz[["_asin"] + amz_cols],
            how="left",
            left_on="ASIN",
            right_on="_asin"
        ).drop(columns=["_asin"])

        # Fill any missing Amazon fields with "not found"
        for c in amz_cols:
            df_merge[c] = df_merge[c].where(df_merge[c].notna(), "not found")

        # ===== Price math =====
        # 1) Parse numeric price from sheet
        df_merge["SheetPrice"] = df_merge[SHEET_PRICE_COL].apply(parse_money)

        # 2) AdjustedPrice = Current price - 11% - 0.30
        def calc_adjusted(x):
            if x is None:
                return None, None
            adj = round((x * 0.89 - 0.30), 2)
            formula = f"{x} - 11% - 0.30 = {adj}"
            return adj, formula

        df_merge[["AdjustedPrice","AdjustedPriceFormula"]] = df_merge["SheetPrice"].apply(
            lambda x: pd.Series(calc_adjusted(x))
        )

        # 3) Parse numeric AmazonPrice
        df_merge["AmazonPrice"] = df_merge["price"].apply(parse_money)

        # 4) Comparison: Amazon price greater than adjusted?
        def cmp(amz, adj):
            if amz is None or adj is None:
                return "not found"
            return bool(amz > adj)

        df_merge["AmazonHigherThanAdjusted"] = df_merge.apply(
            lambda r: cmp(r["AmazonPrice"], r["AdjustedPrice"]), axis=1
        )

        st.success("Done! Preview:")
        preview_cols = [
            "Custom label (SKU)", "ASIN", SHEET_PRICE_COL, "SheetPrice",
            "AdjustedPrice", "AdjustedPriceFormula",
            "AmazonPrice", "AmazonHigherThanAdjusted",
            "TITLE", "Brand", "inStock", "isPrime"
        ]
        existing_cols = [c for c in preview_cols if c in df_merge.columns]
        st.dataframe(df_merge[existing_cols].head(20), use_container_width=True)

        # Download enriched CSV (all columns)
        ts_now = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"{os.path.splitext(upl.name)[0]}_ENRICHED_{ts_now}.csv"
        csv_bytes = df_merge.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download enriched CSV",
            data=csv_bytes,
            file_name=out_name,
            mime="text/csv",
            use_container_width=True
        )
