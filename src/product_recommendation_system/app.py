import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from user_manager import UserManager
import warnings, os
 
warnings.filterwarnings("ignore")
 
st.set_page_config(
    page_title="ShopSmart",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
CSV_PATH     = os.path.join(PROJECT_ROOT, "processed_data", "Raw_Processed_Data.csv")
MODEL_DIR    = BASE_DIR
 
# ── Constants ─────────────────────────────────────────────────────────────────
PRICE_BINS   = [0, 30, 60, 100, 200, 500, float("inf")]
PRICE_LABELS = ["budget", "low", "mid", "mid-high", "high", "premium"]
ADJACENT_BUCKETS = {
    "budget"  : ["budget", "low"],
    "low"     : ["budget", "low", "mid"],
    "mid"     : ["low", "mid", "mid-high"],
    "mid-high": ["mid", "mid-high", "high"],
    "high"    : ["mid-high", "high", "premium"],
    "premium" : ["high", "premium"],
}
ALL_CATEGORIES = [
    "housewares", "sports_leisure", "computers_accessories",
    "bed_bath_table", "health_beauty", "furniture_decor",
    "watches_gifts", "telephony", "toys", "auto",
    "cool_stuff", "garden_tools", "perfumery", "books_general_interest",
    "electronics", "fashion_bags_accessories", "luggage_accessories",
    "stationery", "pet_shop", "food_drink"
]
SEGMENT_META = {
    "A": {"label":"New User",    "color":"#e1f5ee","text":"#085041","emoji":"🆕","desc":"0 interactions"},
    "B": {"label":"Cold-start",  "color":"#e6f1fb","text":"#0c447c","emoji":"❄️","desc":"1 interaction"},
    "C": {"label":"Warm User",   "color":"#faeeda","text":"#633806","emoji":"🌤️","desc":"2-4 interactions"},
    "D": {"label":"Active User", "color":"#eeedfe","text":"#3c3489","emoji":"🔥","desc":"5+ interactions"},
}
 
st.markdown("""
<style>
.seg-badge{display:inline-block;padding:4px 14px;border-radius:20px;font-size:13px;font-weight:600;}
.metric-card{border:1px solid #e9ecef;border-radius:10px;padding:16px;text-align:center;}
.metric-val{font-size:26px;font-weight:700;}
.metric-lbl{font-size:12px;color:#888;margin-top:4px;}
.rec-card{border:1px solid #e9ecef;border-radius:10px;padding:14px 16px;margin-bottom:8px;}
.cat-tag{display:inline-block;background:#f1f3f5;border-radius:6px;padding:2px 8px;font-size:12px;}
</style>
""", unsafe_allow_html=True)
 
 
# ── Init ──────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_um():
    return UserManager()
 
@st.cache_resource(show_spinner="Model load ho raha hai...")
def load_artifacts():
    arts = {}
    try:
        arts["ncf_model"] = keras.models.load_model(
            os.path.join(MODEL_DIR, "hybrid_ncf_best.keras")
        )
    except Exception:
        arts["ncf_model"] = None
    try:
        ue = LabelEncoder()
        ue.classes_ = np.load(os.path.join(MODEL_DIR, "hybrid_user_enc.npy"), allow_pickle=True)
        ie = LabelEncoder()
        ie.classes_ = np.load(os.path.join(MODEL_DIR, "hybrid_item_enc.npy"), allow_pickle=True)
        arts["user_enc"] = ue
        arts["item_enc"] = ie
    except Exception:
        arts["user_enc"] = None
        arts["item_enc"] = None
    return arts
 
@st.cache_data(show_spinner="Dataset load ho raha hai...")
def load_engine(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df["order_status"] == "delivered"]
    full_df = df[[
        "customer_unique_id","product_id","review_score",
        "product_category_name_english","price","order_purchase_timestamp"
    ]].dropna(subset=["customer_unique_id","product_id","review_score"])
    full_df["price_bucket"] = pd.cut(full_df["price"], bins=PRICE_BINS, labels=PRICE_LABELS)
 
    ps = (
        full_df.dropna(subset=["product_category_name_english"])
        .groupby(["product_id","product_category_name_english"])
        .agg(avg_score=("review_score","mean"), num_ratings=("review_score","count"),
             avg_price=("price","mean"),
             price_bucket=("price_bucket", lambda x: x.mode()[0] if len(x)>0 else None))
        .reset_index()
    )
    gm = full_df["review_score"].mean()
    C  = ps["num_ratings"].mean()
    ps["bayesian_score"] = (C*gm + ps["avg_score"]*ps["num_ratings"]) / (C + ps["num_ratings"])
 
    cpr, cr = {}, {}
    for (cat, bkt), grp in ps.groupby(["product_category_name_english","price_bucket"]):
        cpr[(cat, str(bkt))] = grp.sort_values("bayesian_score", ascending=False)["product_id"].tolist()
    for cat, grp in ps.groupby("product_category_name_english"):
        cr[cat] = grp.sort_values("bayesian_score", ascending=False)["product_id"].tolist()
    gt  = ps.sort_values("bayesian_score", ascending=False)["product_id"].tolist()
    cl  = ps.set_index("product_id")["product_category_name_english"].to_dict()
    pl  = ps.set_index("product_id")["avg_price"].to_dict()
    usm = full_df.groupby("customer_unique_id")["product_id"].apply(set).to_dict()
 
    # Popularity boost
    from sklearn.preprocessing import LabelEncoder as _LE
    _n = full_df.copy()
    _uc = _n.groupby("customer_unique_id")["product_id"].count()
    _ic = _n.groupby("product_id")["customer_unique_id"].count()
    _n  = _n[_n["customer_unique_id"].isin(_uc[_uc>=3].index) &
              _n["product_id"].isin(_ic[_ic>=3].index)].copy()
    _ie = _LE(); _n["item_idx"] = _ie.fit_transform(_n["product_id"])
    _p  = np.zeros(_n["item_idx"].nunique())
    _ic2 = _n["item_idx"].value_counts()
    _lc  = np.log1p(_ic2.values); _p[_ic2.index] = _lc/_lc.max()
    pop  = {pid: float(_p[i]) for i,pid in enumerate(_ie.classes_) if i<len(_p)}
 
    return {"ps":ps,"cpr":cpr,"cr":cr,"gt":gt,"cl":cl,"pl":pl,"usm":usm,"pop":pop}
 
 
def get_cands(cat, price, seen, n, log, eng):
    cands = [p for p in eng["cpr"].get((cat,price),[]) if p not in seen]
    if len(cands)>=n: log.append(f"cat+price({price})"); return cands[:n]
    for b in ADJACENT_BUCKETS.get(price,[price]):
        cands += [p for p in eng["cpr"].get((cat,b),[]) if p not in seen and p not in cands]
    if len(cands)>=n: log.append("adj_price"); return cands[:n]
    cands += [p for p in eng["cr"].get(cat,[]) if p not in seen and p not in cands]
    if len(cands)>=n: log.append("cat_only"); return cands[:n]
    log.append("global"); cands += [p for p in eng["gt"] if p not in seen and p not in cands]
    return cands[:n]
 
def diversity(recs, cl, gt, seen=None, k=3, n=10):
    seen = seen or set()
    cats, div, rem = set(), [], []
    for p in recs:
        c = cl.get(p)
        if c and c not in cats and len(cats)<k: div.append(p); cats.add(c)
        else: rem.append(p)
    if len(cats)<k:
        for p in gt:
            if len(cats)>=k: break
            c = cl.get(p)
            if c and c not in cats and p not in seen and p not in div:
                div.append(p); cats.add(c)
    return (div+rem)[:n]
 
def recommend(uid, seg, eng, arts, um_obj, n=10, ob_cats=None, ob_price=None):
    cl  = eng["cl"]; pl = eng["pl"]; ps = eng["ps"]
    gt  = eng["gt"]; pop = eng["pop"]
    seen = eng["usm"].get(uid, set())
 
    if seg=="A":
        cats  = ob_cats or []
        price = ob_price or "mid"
        if not cats:
            recs = [p for p in gt if p not in seen][:n]
            strat = "Global top-rated"
        else:
            recs, log = [], []
            slots = max(2, n//len(cats))
            for cat in cats:
                recs += get_cands(cat, price, seen|set(recs), slots, log, eng)
            if len(recs)<n:
                recs += [p for p in gt if p not in seen and p not in recs][:n-len(recs)]
            recs  = diversity(recs, cl, gt, seen, n=n)
            strat = "Onboarding: category + price"
        return _res(recs[:n], ps, pl, cl, seg, strat, price, cats)
 
    if seg=="D":
        nm = arts.get("ncf_model"); ue = arts.get("user_enc"); ie = arts.get("item_enc")
        if nm and ue and ie and uid in ue.classes_:
            ui   = ue.transform([uid])[0]
            alls = np.arange(len(ie.classes_))
            si   = {ie.transform([p])[0] for p in seen if p in ie.classes_}
            cand = np.array([i for i in alls if i not in si])
            sc   = nm.predict([np.full(len(cand),ui), cand], batch_size=512, verbose=0).flatten()
            cp   = ie.inverse_transform(cand)
            sc  += 0.1 * np.array([pop.get(p,0.0) for p in cp])
            ti   = np.argsort(sc)[::-1][:n*3]
            recs = diversity(list(ie.inverse_transform(cand[ti])), cl, gt, seen, n=n)
            pb   = um_obj.get_user_price_pref(uid) or "mid"
            cats = list({cl.get(p) for p in recs if cl.get(p)})
            return _res(recs[:n], ps, pl, cl, seg, "NCF + popularity boost", pb, cats)
 
    cats  = um_obj.get_user_category_prefs(uid) or (ob_cats or [])
    price = um_obj.get_user_price_pref(uid) or (ob_price or "mid")
    if not cats:
        recs = [p for p in gt if p not in seen][:n]
        return _res(recs, ps, pl, cl, seg, "Global top-rated", "mid", ["global"])
 
    slots = [max(1, round(n/len(cats)))] * len(cats)
    slots[0] += n - sum(slots)
    recs, used, log = [], [], []
    for cat, s in zip(cats, slots):
        if s<=0: continue
        recs += get_cands(cat, price, seen|set(recs), s, log, eng)
        used.append(cat)
    if len(recs)<n:
        recs += [p for p in gt if p not in seen and p not in recs][:n-len(recs)]
    recs  = diversity(recs, cl, gt, seen, n=n)
    strat = "Category + price: " + " → ".join(dict.fromkeys(log))
    return _res(recs[:n], ps, pl, cl, seg, strat, price, used)
 
def _res(recs, ps, pl, cl, seg, strat, pb, cats):
    rows = []
    for i,pid in enumerate(recs,1):
        r = ps[ps["product_id"]==pid]
        if not r.empty:
            r = r.iloc[0]
            rows.append({"rank":i,"product_id":pid,
                "category":r.get("product_category_name_english",cl.get(pid,"—")),
                "bayesian_score":round(float(r.get("bayesian_score",0)),3),
                "avg_price":round(float(r.get("avg_price",0)),2),
                "num_ratings":int(r.get("num_ratings",0))})
        else:
            rows.append({"rank":i,"product_id":pid,"category":cl.get(pid,"—"),
                         "bayesian_score":0,"avg_price":0,"num_ratings":0})
    return {"segment":seg,"strategy":strat,"price_bucket":pb,
            "categories_used":cats,"recommendations":pd.DataFrame(rows)}
 
def sc_color(s):
    if s>=4.7: return "#1d9e75"
    if s>=4.4: return "#639922"
    if s>=4.0: return "#ba7517"
    return "#d85a30"
 
 
# ── Session state ─────────────────────────────────────────────────────────────
for k,v in [("logged_in",False),("current_user",None),
            ("last_result",None),("engine_loaded",False)]:
    if k not in st.session_state:
        st.session_state[k] = v
 
um        = get_um()
artifacts = load_artifacts()
 
 
# ════════════════════════════════════════════════════════════════════════════
#  LOGIN / REGISTER PAGE
# ════════════════════════════════════════════════════════════════════════════
if not st.session_state.logged_in:
 
    st.markdown("<h1 style='text-align:center;'>🛍️ ShopSmart</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#888;'>Personalized Product Recommendations</p>",
                unsafe_allow_html=True)
    st.divider()
 
    _, col, _ = st.columns([1,2,1])
    with col:
        tab_l, tab_r = st.tabs(["🔑 Login", "📝 Register"])
 
        # ── LOGIN TAB ──────────────────────────────────────────────────────
        with tab_l:
            st.subheader("Login karo")
            lemail = st.text_input("Email", placeholder="aapka@email.com", key="login_email")
            lpass  = st.text_input("Password", type="password", key="login_pass")
            if st.button("Login", type="primary", use_container_width=True):
                if not lemail.strip():
                    st.warning("Email daalo.")
                elif not lpass.strip():
                    st.warning("Password daalo.")
                else:
                    res = um.login_user(lemail.strip(), lpass.strip())
                    if res["status"] == "success":
                        st.session_state.logged_in    = True
                        st.session_state.current_user = res["user"]
                        st.rerun()
                    elif res["status"] == "wrong_password":
                        st.error("❌ Password galat hai.")
                    else:
                        st.error("❌ " + res["message"])
 
        # ── REGISTER TAB ───────────────────────────────────────────────────
        with tab_r:
            st.subheader("Naya Account Banao")
            rname  = st.text_input("Naam *", key="rname")
            remail = st.text_input("Email *", placeholder="aapka@email.com", key="remail")
            rpass  = st.text_input("Password *", type="password", key="rpass")
            rpass2 = st.text_input("Password Confirm karo *", type="password", key="rpass2")
            rcats  = st.multiselect(
                "Pasandida Categories * (max 3)",
                ALL_CATEGORIES, max_selections=3,
                format_func=lambda x: x.replace("_"," ").title(),
                key="rcats"
            )
            rprice = st.selectbox(
                "Budget *", PRICE_LABELS, index=3,
                format_func=lambda x: {
                    "budget":"Budget (< R$30)","low":"Low (R$30-60)",
                    "mid":"Mid (R$60-100)","mid-high":"Mid-High (R$100-200)",
                    "high":"High (R$200-500)","premium":"Premium (R$500+)"
                }.get(x,x), key="rprice"
            )
            if st.button("Register", type="primary", use_container_width=True):
                if not rname.strip():
                    st.error("Naam daalo.")
                elif not remail.strip():
                    st.error("Email daalo.")
                elif not rpass.strip():
                    st.error("Password daalo.")
                elif rpass.strip() != rpass2.strip():
                    st.error("❌ Dono passwords match nahi kar rahe!")
                elif len(rpass.strip()) < 6:
                    st.error("Password kam se kam 6 characters ka hona chahiye.")
                elif not rcats:
                    st.error("Kam se kam 1 category select karo.")
                else:
                    res = um.register_user(rname, remail, rpass, rcats, rprice)
                    if res["status"] == "success":
                        st.success("✅ " + res["message"])
                        st.balloons()
                    elif res["status"] == "exists":
                        st.warning("⚠️ " + res["message"])
 
    st.stop()
 
 
# ════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ════════════════════════════════════════════════════════════════════════════
user    = st.session_state.current_user
uid     = user["user_id"]
seg     = um.get_user_segment(uid)
meta    = SEGMENT_META[seg]
 
# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"### 👋 {user['name']}")
    st.markdown(
        f'<span class="seg-badge" style="background:{meta["color"]};color:{meta["text"]}">'
        f'{meta["emoji"]} Segment {seg} — {meta["label"]}</span>',
        unsafe_allow_html=True
    )
    st.caption(f"ID: `{uid}`")
    st.caption(f"Email: {user.get('email','—')}")
    st.divider()
 
    total = int(user.get("total_interactions",0))
    st.markdown(f"**Total Interactions:** {total}")
    needed_map = {"A":1,"B":1,"C":max(0,5-total),"D":0}
    next_map   = {"A":"B","B":"C","C":"D","D":"D"}
    if seg!="D":
        st.caption(f"Segment {next_map[seg]} ke liye **{needed_map[seg]}** aur interaction chahiye")
        max_val = {"A":1,"B":2,"C":5,"D":5}[seg]
        st.progress(min(total/max(max_val,1), 1.0))
 
    st.divider()
    top_n = st.slider("Recommendations", 3, 20, 10)
    st.divider()
    if st.button("🚪 Logout"):
        for k in ["logged_in","current_user","last_result","engine_loaded"]:
            st.session_state[k] = False if k=="logged_in" or k=="engine_loaded" else None
        st.rerun()
 
# ── Load engine ───────────────────────────────────────────────────────────────
if not st.session_state.engine_loaded:
    if os.path.exists(CSV_PATH):
        with st.spinner("Data load ho raha hai..."):
            st.session_state["engine"] = load_engine(CSV_PATH)
            st.session_state.engine_loaded = True
    else:
        st.error(f"CSV nahi mili: `{CSV_PATH}`")
        st.stop()
 
engine = st.session_state["engine"]
 
st.title("🛍️ ShopSmart")
 
tab_rec, tab_hist, tab_prof = st.tabs(["🎯 Recommendations","📋 History","👤 Profile"])
 
 
# ── TAB 1: RECOMMENDATIONS ────────────────────────────────────────────────────
with tab_rec:
    cl, cr = st.columns([1,2])
 
    with cl:
        st.markdown(
            f'<span class="seg-badge" style="background:{meta["color"]};color:{meta["text"]};font-size:14px;">'
            f'{meta["emoji"]} {meta["label"]} · {meta["desc"]}</span>',
            unsafe_allow_html=True
        )
        st.markdown(" ")
 
        if seg=="A":
            st.markdown("**Preferences select karo:**")
            ob_cats = st.multiselect(
                "Categories", ALL_CATEGORIES,
                default=um.get_user_category_prefs(uid) or [],
                max_selections=3,
                format_func=lambda x: x.replace("_"," ").title(),
                key="ob_cats"
            )
            ob_price = st.selectbox(
                "Budget", PRICE_LABELS,
                index=PRICE_LABELS.index(um.get_user_price_pref(uid) or "mid"),
                key="ob_price"
            )
        else:
            ob_cats  = um.get_user_category_prefs(uid)
            ob_price = um.get_user_price_pref(uid)
            cats_str = ", ".join(ob_cats) if ob_cats else "—"
            st.caption(f"**Categories:** {cats_str}")
            st.caption(f"**Budget:** {ob_price}")
 
        st.divider()
        go = st.button("🎯 Recommend Karo", type="primary", use_container_width=True)
 
    with cr:
        if go:
            with st.spinner("Recommendations ban rahi hain..."):
                result = recommend(uid, seg, engine, artifacts, um,
                                   n=top_n, ob_cats=ob_cats, ob_price=ob_price)
                st.session_state.last_result = result
 
        if st.session_state.last_result:
            res = st.session_state.last_result
            df  = res["recommendations"]
            st.caption(f"**Strategy:** {res['strategy']}")
 
            for _, row in df.iterrows():
                rank  = int(row["rank"])
                pid   = str(row["product_id"])
                cat   = str(row["category"]) if pd.notna(row["category"]) else "—"
                score = float(row["bayesian_score"]) if pd.notna(row["bayesian_score"]) else 0
                price = float(row["avg_price"])      if pd.notna(row["avg_price"])      else 0
                nrat  = int(row["num_ratings"])       if pd.notna(row["num_ratings"])    else 0
 
                c1, c2 = st.columns([6,1])
                with c1:
                    st.markdown(f"""
                    <div class="rec-card">
                      <div style="display:flex;align-items:center;gap:12px;">
                        <div style="font-size:20px;font-weight:700;color:#ccc;min-width:30px;">#{rank}</div>
                        <div style="flex:1;">
                          <span class="cat-tag">{cat.replace("_"," ")}</span>
                          <span style="font-size:12px;color:#aaa;margin-left:8px;">
                            R$ {price:.0f} · {nrat} ratings
                          </span>
                          <div style="font-size:11px;color:#bbb;font-family:monospace;margin-top:4px;">
                            {pid[:30]}...
                          </div>
                        </div>
                        <div style="text-align:right;">
                          <div style="font-size:20px;font-weight:700;color:{sc_color(score)};">
                            {score:.3f}
                          </div>
                          <div style="font-size:10px;color:#aaa;">Bayesian</div>
                        </div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
                with c2:
                    st.markdown("<div style='padding-top:10px;'>", unsafe_allow_html=True)
                    if st.button("👁️", key=f"v_{pid}_{rank}", help="Track karo"):
                        um.log_interaction(uid, pid, cat, price, "click")
                        st.session_state.current_user = um.get_user(uid)
                        user = st.session_state.current_user
                        new_seg = um.get_user_segment(uid)
                        if new_seg != seg:
                            st.toast(f"🎉 Segment upgrade! {seg} → {new_seg}", icon="🎉")
                        else:
                            st.toast("✅ Interaction saved!", icon="✅")
                        st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)
 
            st.download_button(
                "⬇️ CSV Download", df.to_csv(index=False),
                file_name=f"recs_{uid}.csv", mime="text/csv"
            )
        else:
            st.markdown("""
            <div style='text-align:center;padding:60px 20px;color:#aaa;'>
                <div style='font-size:48px;'>🛍️</div>
                <div style='margin-top:12px;'>Recommend Karo button click karo</div>
            </div>
            """, unsafe_allow_html=True)
 
 
# ── TAB 2: HISTORY ────────────────────────────────────────────────────────────
with tab_hist:
    st.subheader("Meri Interaction History")
    ints = um.get_user_interactions(uid)
 
    if ints.empty:
        st.info("Abhi koi interaction nahi. Recommendations mein 👁️ click karo!")
    else:
        c1,c2,c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{len(ints)}</div><div class="metric-lbl">Total Interactions</div></div>', unsafe_allow_html=True)
        with c2:
            nc = ints["category"].nunique() if "category" in ints.columns else 0
            st.markdown(f'<div class="metric-card"><div class="metric-val">{nc}</div><div class="metric-lbl">Categories Explored</div></div>', unsafe_allow_html=True)
        with c3:
            cs = um.get_user_segment(uid)
            st.markdown(f'<div class="metric-card"><div class="metric-val">{SEGMENT_META[cs]["emoji"]} {cs}</div><div class="metric-lbl">Current Segment</div></div>', unsafe_allow_html=True)
 
        st.divider()
        show = [c for c in ["timestamp","product_id","category","price","action_type"] if c in ints.columns]
        st.dataframe(ints[show], use_container_width=True, hide_index=True)
 
 
# ── TAB 3: PROFILE ────────────────────────────────────────────────────────────
with tab_prof:
    st.subheader("Mera Profile")
    fu = um.get_user(uid)
    if fu:
        c1,c2 = st.columns(2)
        with c1:
            st.markdown("**Account Info**")
            st.markdown(f"- **Naam:** {fu.get('name','—')}")
            st.markdown(f"- **Email:** {fu.get('email','—')}")
            st.markdown(f"- **User ID:** `{fu.get('user_id','—')}`")
            st.markdown(f"- **Joined:** {fu.get('joined_date','—')}")
            st.markdown(f"- **Interactions:** {fu.get('total_interactions',0)}")
        with c2:
            s = fu.get("segment","A")
            m = SEGMENT_META[s]
            st.markdown("**Segment**")
            st.markdown(
                f'<span class="seg-badge" style="background:{m["color"]};color:{m["text"]}">'
                f'{m["emoji"]} {s} — {m["label"]}</span>',
                unsafe_allow_html=True
            )
            st.markdown(f"\n- **Categories:** {fu.get('category_pref','—')}")
            st.markdown(f"- **Price Pref:** {fu.get('price_pref','—')}")
 
        st.divider()
        st.markdown("**Preferences Update Karo**")
        nc = st.multiselect(
            "Categories", ALL_CATEGORIES,
            default=um.get_user_category_prefs(uid),
            max_selections=3,
            format_func=lambda x: x.replace("_"," ").title(),
            key="uc"
        )
        np_sel = st.selectbox(
            "Budget", PRICE_LABELS,
            index=PRICE_LABELS.index(um.get_user_price_pref(uid) or "mid"),
            key="up"
        )
        if st.button("Update", type="primary"):
            if nc:
                um.update_user_prefs(uid, nc, np_sel)
                st.session_state.current_user = um.get_user(uid)
                st.success("Updated!")
                st.rerun()
            else:
                st.warning("Kam se kam 1 category select karo.")