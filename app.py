Y
import streamlit as st
import numpy as np
import sympy as sp
import pandas as pd
import matplotlib.pyplot as plt
 
st.set_page_config(
    page_title="WolfeX · Optimizador",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)
 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
 
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.stApp { background: #07090f; }
.block-container { padding: 0 !important; max-width: 100% !important; }
 
/* ── TOPBAR ── */
.topbar {
    background: rgba(255,255,255,0.03);
    border-bottom: 1px solid rgba(255,255,255,0.06);
    padding: 1rem 2.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    backdrop-filter: blur(12px);
}
.topbar-logo {
    font-size: 1.4rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.topbar-nav { display: flex; gap: 6px; }
.nav-chip {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    color: #94a3b8;
    border-radius: 8px;
    padding: 6px 14px;
    font-size: 0.8rem;
    font-weight: 500;
}
.nav-chip.active {
    background: rgba(139,92,246,0.2);
    border-color: rgba(139,92,246,0.4);
    color: #a78bfa;
}
 
/* ── LAYOUT ── */
.app-layout {
    display: grid;
    grid-template-columns: 380px 1fr;
    min-height: calc(100vh - 65px);
}
.left-panel {
    background: #0d1117;
    border-right: 1px solid rgba(255,255,255,0.06);
    padding: 2rem 1.5rem;
    display: flex;
    flex-direction: column;
    gap: 1.2rem;
}
.right-panel {
    padding: 2rem 2.5rem;
    overflow-y: auto;
}
 
/* ── INPUT CARDS ── */
.input-section {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 1.2rem;
}
.input-section-title {
    font-size: 0.68rem;
    font-weight: 600;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 6px;
}
 
/* ── METHOD BUTTONS ── */
.method-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
    margin-top: 0.5rem;
}
.method-btn {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 10px 6px;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s;
    color: #64748b;
    font-size: 0.75rem;
    font-weight: 500;
}
.method-btn:hover { border-color: rgba(139,92,246,0.4); color: #a78bfa; }
.method-btn.sel {
    background: rgba(139,92,246,0.15);
    border-color: rgba(139,92,246,0.5);
    color: #c4b5fd;
}
.method-btn .icon { font-size: 1.3rem; display: block; margin-bottom: 4px; }
 
/* ── RUN BUTTON ── */
.run-btn {
    width: 100%;
    background: linear-gradient(135deg, #7c3aed, #6366f1);
    border: none;
    border-radius: 12px;
    color: white;
    font-size: 1rem;
    font-weight: 700;
    padding: 14px;
    cursor: pointer;
    letter-spacing: 0.02em;
    box-shadow: 0 4px 24px rgba(124,58,237,0.4);
    transition: all 0.2s;
    margin-top: 0.5rem;
}
 
/* ── EXAMPLE PILLS ── */
.ex-pills { display: flex; flex-direction: column; gap: 6px; }
.ex-pill {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 8px;
    padding: 8px 12px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #34d399;
    cursor: pointer;
    transition: all 0.15s;
}
.ex-pill:hover { border-color: #34d399; background: rgba(52,211,153,0.05); }
.ex-pill-label { font-family: 'Inter', sans-serif; color: #475569; font-size: 0.7rem; margin-top: 2px; }
 
/* ── HERO RIGHT ── */
.hero-right {
    text-align: center;
    padding: 5rem 2rem;
}
.hero-glow {
    font-size: 4rem;
    font-weight: 800;
    line-height: 1.1;
    letter-spacing: -2px;
    background: linear-gradient(135deg, #ffffff 0%, #a78bfa 50%, #60a5fa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 1rem;
}
.hero-desc {
    color: #475569;
    font-size: 1.05rem;
    max-width: 480px;
    margin: 0 auto 2rem;
    line-height: 1.7;
}
.feature-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    max-width: 560px;
    margin: 0 auto;
}
.feature-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 1.2rem 1rem;
    text-align: center;
}
.feature-card .fi { font-size: 1.6rem; margin-bottom: 6px; }
.feature-card .ft { font-size: 0.78rem; color: #64748b; }
.feature-card .fv { font-size: 0.9rem; font-weight: 600; color: #e2e8f0; }
 
/* ── RESULT AREA ── */
.result-topbar {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 1.5rem;
}
.result-badge {
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.8rem;
    font-weight: 600;
}
.badge-ok   { background: rgba(52,211,153,0.15); color: #34d399; border: 1px solid rgba(52,211,153,0.3); }
.badge-warn { background: rgba(251,191,36,0.15);  color: #fbbf24; border: 1px solid rgba(251,191,36,0.3); }
 
.kpi-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    margin-bottom: 1.5rem;
}
.kpi {
    background: #0d1117;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.3rem 1.5rem;
    position: relative;
    overflow: hidden;
}
.kpi::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
}
.kpi.v::after  { background: linear-gradient(90deg,#7c3aed,#a78bfa); }
.kpi.f::after  { background: linear-gradient(90deg,#059669,#34d399); }
.kpi.it::after { background: linear-gradient(90deg,#2563eb,#60a5fa); }
.kpi-label { font-size: 0.7rem; color: #475569; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px; }
.kpi-value { font-family: 'JetBrains Mono', monospace; font-size: 1.4rem; font-weight: 600; }
.kpi.v  .kpi-value { color: #c4b5fd; }
.kpi.f  .kpi-value { color: #34d399; }
.kpi.it .kpi-value { color: #60a5fa; }
 
.info-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-bottom: 1.5rem;
}
.info-block {
    background: #0d1117;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 1rem 1.2rem;
}
.info-block .ib-label { font-size: 0.7rem; color: #475569; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px; }
.info-block .ib-val { font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #94a3b8; word-break: break-all; }
 
.section-hdr {
    font-size: 0.75rem;
    font-weight: 600;
    color: #334155;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 1.5rem 0 0.8rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-hdr::after { content:''; flex:1; height:1px; background:rgba(255,255,255,0.05); }
 
/* Streamlit overrides */
section[data-testid="stSidebar"] { display: none; }
.stTextInput input, .stNumberInput input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    font-family: 'JetBrains Mono', monospace !important;
}
.stTextInput label, .stNumberInput label, .stSelectbox label {
    color: #475569 !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)
 
 
# ──────────────────────────────────────────────
# MATEMÁTICA
# ──────────────────────────────────────────────
def parse_vector(text, n):
    try:
        vals = [float(v.strip()) for v in text.replace(";",",").split(",") if v.strip()]
        if len(vals) != n: raise ValueError(f"Se necesitan {n} valores.")
        return np.array(vals, dtype=float)
    except Exception as e:
        raise ValueError(f"Punto inválido: {e}")
 
def build_functions(expr_text, n):
    syms = sp.symbols(f"x1:{n+1}")
    ld = {f"x{i+1}": syms[i] for i in range(n)}
    ld.update({"sin":sp.sin,"cos":sp.cos,"tan":sp.tan,"exp":sp.exp,"log":sp.log,
               "ln":sp.log,"sqrt":sp.sqrt,"pi":sp.pi,"e":sp.E,"E":sp.E,
               "abs":sp.Abs,"Abs":sp.Abs,"sinh":sp.sinh,"cosh":sp.cosh,
               "tanh":sp.tanh,"atan":sp.atan,"asin":sp.asin,"acos":sp.acos})
    try:
        expr = sp.sympify(expr_text.replace("^","**"), locals=ld)
    except Exception as e:
        raise ValueError(f"No se pudo interpretar: {e}")
    unk = expr.free_symbols - set(syms)
    if unk: raise ValueError(f"Variables no permitidas: {', '.join(str(s) for s in unk)}")
    ge = [sp.diff(expr,v) for v in syms]
    he = sp.hessian(expr, syms)
    fl = sp.lambdify((syms,), expr, "numpy")
    gl = sp.lambdify((syms,), ge, "numpy")
    hl = sp.lambdify((syms,), he, "numpy")
    def f(x):
        v = float(np.asarray(fl(tuple(x)), dtype=float))
        if not np.isfinite(v): raise FloatingPointError("f no finita")
        return v
    def g(x):
        gv = np.array(gl(tuple(x)), dtype=float).reshape(-1)
        if not np.all(np.isfinite(gv)): raise FloatingPointError("grad no finito")
        return gv
    def h(x): return np.array(hl(tuple(x)), dtype=float)
    return expr, f, g, h
 
def wolfe_ls(f, g, x, d, a0, c1, c2, rho):
    fx=f(x); gx=g(x); dp=float(np.dot(gx,d))
    if dp>=0: d=-gx; dp=float(np.dot(gx,d))
    a=a0; ba=None; bf=fx
    for _ in range(60):
        xn=x+a*d
        try: fn=f(xn); gn=g(xn)
        except: a*=rho; continue
        if np.isfinite(fn) and fn<bf: bf=fn; ba=a
        if fn<=fx+c1*a*dp and abs(float(np.dot(gn,d)))<=c2*abs(dp): return a,True
        a*=rho
    return (ba,False) if ba else (a,False)
 
def hess_pd(H):
    H=0.5*(H+H.T)
    try: me=float(np.min(np.linalg.eigvalsh(H)))
    except: me=-1.
    if me>1e-8: return H
    return H+(max(0.,-me)+1e-3)*np.eye(H.shape[0])
 
def run_opt(method, f, g, h, x0, max_iter, tol, a0, c1, c2, rho):
    x=x0.copy().astype(float); hist=[]; dp=None; gp=None
    reason="Máximo de iteraciones alcanzado"
    for k in range(max_iter+1):
        try: fx=f(x); gx=g(x)
        except: reason="Dominio inválido."; break
        err=float(np.linalg.norm(gx))
        hist.append({"iteración":k,"f(x)":fx,"‖∇f‖":err,"x":x.copy()})
        if err<tol: reason="Convergencia: ‖∇f‖ < tolerancia"; break
        if k==max_iter: break
        if method=="Gradiente": d=-gx
        elif method=="Gradiente conjugado":
            if k==0 or dp is None: d=-gx
            else:
                beta=max(0.,float(np.dot(gx,gx-gp)/max(np.dot(gp,gp),1e-12)))
                d=-gx+beta*dp
                if float(np.dot(gx,d))>=0: d=-gx
        else:
            H=hess_pd(h(x))
            try: d=-np.linalg.solve(H,gx)
            except: d=-np.linalg.pinv(H).dot(gx)
            if float(np.dot(gx,d))>=0: d=-gx
        a,_=wolfe_ls(f,g,x,d,a0,c1,c2,rho)
        xn=x+a*d; gp=gx.copy(); dp=d.copy(); x=xn
        if not np.all(np.isfinite(x)): reason="Divergencia."; break
    if not hist: raise ValueError("No se pudo evaluar en el punto inicial.")
    try: ff=f(x)
    except: ff=hist[-1]["f(x)"]; x=hist[-1]["x"]
    return x, ff, len(hist)-1, hist, reason
 
 
# ──────────────────────────────────────────────
# ESTADO
# ──────────────────────────────────────────────
if "method" not in st.session_state:
    st.session_state.method = "Newton"
if "func" not in st.session_state:
    st.session_state.func = "x1**2 + x2**2"
if "x0" not in st.session_state:
    st.session_state.x0 = "3, 4"
if "resultado" not in st.session_state:
    st.session_state.resultado = None
 
 
# ──────────────────────────────────────────────
# TOPBAR
# ──────────────────────────────────────────────
st.markdown("""
<div class="topbar">
  <div class="topbar-logo">⚡ WolfeX</div>
  <div class="topbar-nav">
    <span class="nav-chip active">Optimizador</span>
    <span class="nav-chip">Primer Semestre 2026</span>
    <span class="nav-chip">Wolfe Fuerte</span>
  </div>
</div>
""", unsafe_allow_html=True)
 
 
# ──────────────────────────────────────────────
# LAYOUT: dos columnas manuales
# ──────────────────────────────────────────────
left, right = st.columns([1.1, 2.4])
 
with left:
    st.markdown("#### ⚙️ Configuración")
 
    # Función objetivo
    func = st.text_input("Función f(x)", value=st.session_state.func,
                          help="Usa x1, x2, ..., xn · sin cos exp log sqrt ^ o **")
    st.session_state.func = func
 
    # Ejemplos rápidos
    st.markdown("**Ejemplos rápidos**")
    ec1, ec2, ec3 = st.columns(3)
    if ec1.button("Esférica", use_container_width=True):
        st.session_state.func = "x1**2 + x2**2"
        st.session_state.x0 = "3, 4"
        st.rerun()
    if ec2.button("Desplazada", use_container_width=True):
        st.session_state.func = "(x1-1)**2 + (x2+2)**2"
        st.session_state.x0 = "0, 0"
        st.rerun()
    if ec3.button("Rosenbrock", use_container_width=True):
        st.session_state.func = "100*(x2-x1**2)**2 + (1-x1)**2"
        st.session_state.x0 = "-1.2, 1"
        st.rerun()
 
    st.markdown("---")
 
    # Método — botones visuales
    st.markdown("**Método de optimización**")
    m1, m2, m3 = st.columns(3)
    def mbtn(col, label, icon, key):
        sel = st.session_state.method == label
        style = "background:#2d1b69;border:1.5px solid #7c3aed;color:#c4b5fd;" if sel else ""
        if col.button(f"{icon}\n{label}", key=key, use_container_width=True):
            st.session_state.method = label
            st.rerun()
    mbtn(m1, "Gradiente",  "▽", "m1")
    mbtn(m2, "Gradiente conjugado", "⇉", "m2")
    mbtn(m3, "Newton", "⊕", "m3")
 
    st.markdown(f"✅ Método activo: **{st.session_state.method}**")
    st.markdown("---")
 
    # Parámetros numéricos
    n       = st.number_input("Número de variables", 1, 20, 2, 1)
    x0_txt  = st.text_input("Punto inicial x₀", value=st.session_state.x0)
    st.session_state.x0 = x0_txt
    max_it  = st.number_input("Máx. iteraciones", 1, 100000, 100, 10)
    tol     = st.number_input("Tolerancia ε", 1e-15, 1.0, 1e-6, format="%.1e")
 
    with st.expander("⚙️ Parámetros Wolfe"):
        alpha0 = st.number_input("α inicial", 1e-12, 100.0, 1.0, format="%.4f")
        c1     = st.number_input("c₁ Armijo",  1e-12, 0.5,  1e-4, format="%.1e")
        c2     = st.number_input("c₂ curvatura", 0.01, 0.99, 0.9, format="%.2f")
        rho    = st.number_input("ρ reducción α", 0.01, 0.99, 0.5, format="%.2f")
 
    st.markdown("---")
    run = st.button("⚡  Ejecutar optimización", type="primary", use_container_width=True)
 
    if run:
        try:
            if not (0 < c1 < c2 < 1):
                st.error("Se requiere 0 < c₁ < c₂ < 1")
            else:
                with st.spinner("Calculando..."):
                    expr, f, g, h = build_functions(st.session_state.func, int(n))
                    x0v = parse_vector(st.session_state.x0, int(n))
                    xmin, fmin, iters, history, reason = run_opt(
                        st.session_state.method, f, g, h, x0v,
                        int(max_it), tol, alpha0, c1, c2, rho)
                st.session_state.resultado = {
                    "xmin": xmin, "fmin": fmin, "iters": iters,
                    "history": history, "reason": reason,
                    "expr": sp.sstr(expr), "method": st.session_state.method
                }
        except Exception as ex:
            st.error(f"❌ {ex}")
            st.session_state.resultado = None
 
 
# ──────────────────────────────────────────────
# PANEL DERECHO
# ──────────────────────────────────────────────
with right:
    res = st.session_state.resultado
 
    if res is None:
        # Pantalla de bienvenida
        st.markdown("""
        <div style="padding:5rem 2rem; text-align:center;">
          <div style="font-size:4rem;font-weight:800;letter-spacing:-2px;
               background:linear-gradient(135deg,#fff 0%,#a78bfa 50%,#60a5fa 100%);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            Optimización<br>Numérica
          </div>
          <div style="color:#334155;font-size:1rem;max-width:460px;margin:1.2rem auto 2.5rem;line-height:1.8;">
            Encuentra el mínimo de cualquier función usando
            <strong style="color:#a78bfa">Gradiente</strong>,
            <strong style="color:#60a5fa">Gradiente Conjugado</strong> o
            <strong style="color:#34d399">Newton</strong>,
            con búsqueda de línea que satisface las condiciones de Wolfe.
          </div>
          <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;max-width:520px;margin:0 auto;">
            <div style="background:#0d1117;border:1px solid rgba(255,255,255,0.06);border-radius:14px;padding:1.3rem 1rem;">
              <div style="font-size:1.8rem">∇</div>
              <div style="font-size:0.85rem;font-weight:600;color:#c4b5fd;margin:6px 0 2px">Gradiente</div>
              <div style="font-size:0.72rem;color:#334155">Descenso clásico</div>
            </div>
            <div style="background:#0d1117;border:1px solid rgba(255,255,255,0.06);border-radius:14px;padding:1.3rem 1rem;">
              <div style="font-size:1.8rem">⇉</div>
              <div style="font-size:0.85rem;font-weight:600;color:#60a5fa;margin:6px 0 2px">Conjugado</div>
              <div style="font-size:0.72rem;color:#334155">Polak-Ribière</div>
            </div>
            <div style="background:#0d1117;border:1px solid rgba(255,255,255,0.06);border-radius:14px;padding:1.3rem 1rem;">
              <div style="font-size:1.8rem">⊕</div>
              <div style="font-size:0.85rem;font-weight:600;color:#34d399;margin:6px 0 2px">Newton</div>
              <div style="font-size:0.72rem;color:#334155">Hessiano regularizado</div>
            </div>
          </div>
          <div style="margin-top:3rem;color:#1e293b;font-size:0.85rem;">
            ← Configura los parámetros y presiona <strong style="color:#a78bfa">Ejecutar</strong>
          </div>
        </div>
        """, unsafe_allow_html=True)
 
    else:
        convergio = "Convergencia" in res["reason"]
        bstyle = "background:rgba(52,211,153,.12);color:#34d399;border:1px solid rgba(52,211,153,.3);" if convergio \
                 else "background:rgba(251,191,36,.12);color:#fbbf24;border:1px solid rgba(251,191,36,.3);"
        bicon = "✅" if convergio else "⚠️"
 
        st.markdown(f"""
        <div style="border-radius:10px;padding:10px 16px;font-size:0.88rem;
             font-weight:600;margin-bottom:1.2rem;{bstyle}">
          {bicon} &nbsp;{res['reason']}
        </div>""", unsafe_allow_html=True)
 
        # KPIs
        xstr = np.array2string(res["xmin"], precision=5, suppress_small=True)
        c1k, c2k, c3k = st.columns(3)
        with c1k:
            st.metric("📍 Punto mínimo x*", xstr)
        with c2k:
            st.metric("🎯 Valor f(x*)", f"{res['fmin']:.6g}")
        with c3k:
            st.metric("🔁 Iteraciones", res["iters"])
 
        fe = res["history"][-1]["‖∇f‖"]
        i1, i2 = st.columns(2)
        i1.info(f"**Error final ‖∇f‖:** `{fe:.4e}`")
        i2.info(f"**Función:** `{res['expr']}`")
 
        # Tabs para resultados
        tab1, tab2 = st.tabs(["📈 Convergencia", "📊 Historial"])
 
        with tab1:
            errs = np.array([h["‖∇f‖"] for h in res["history"]])
            its  = np.array([h["iteración"] for h in res["history"]])
            pos  = errs[errs>0]
            use_log = len(pos)>=2 and errs.max()/pos.min()>100
 
            colors = {"Gradiente":"#a78bfa","Gradiente conjugado":"#60a5fa","Newton":"#34d399"}
            col = colors.get(res["method"], "#a78bfa")
 
            fig, ax = plt.subplots(figsize=(10,4))
            fig.patch.set_facecolor("#07090f")
            ax.set_facecolor("#0d1117")
 
            y = np.maximum(errs,1e-16)
            if use_log:
                ax.semilogy(its, y, color=col, lw=2.5, marker="o", ms=5,
                            markerfacecolor="#07090f", markeredgecolor=col, markeredgewidth=2,
                            label=res["method"])
                ax.fill_between(its, 1e-16, y, alpha=0.07, color=col)
                ax.set_ylabel("‖∇f‖  (log)", color="#475569", fontsize=10)
            else:
                ax.plot(its, errs, color=col, lw=2.5, marker="o", ms=5,
                        markerfacecolor="#07090f", markeredgecolor=col, markeredgewidth=2,
                        label=res["method"])
                ax.fill_between(its, 0, errs, alpha=0.07, color=col)
                ax.set_ylabel("‖∇f‖", color="#475569", fontsize=10)
 
            ax.set_xlabel("Iteración", color="#475569", fontsize=10)
            ax.set_title("Error versus número de iteraciones", color="#e2e8f0", fontsize=12, fontweight="bold")
            ax.tick_params(colors="#334155", labelsize=8)
            for sp2 in ax.spines.values(): sp2.set_color("#1e293b")
            ax.grid(True, alpha=0.3, color="#1e293b", linestyle="--")
            ax.legend(fontsize=9, framealpha=0.1, labelcolor=col)
            fig.tight_layout()
            st.pyplot(fig)
 
        with tab2:
            df = pd.DataFrame([{
                "it.": h["iteración"],
                "f(x)": round(h["f(x)"],8),
                "‖∇f‖": round(h["‖∇f‖"],8),
                "x": np.array2string(h["x"],precision=4,suppress_small=True)
            } for h in res["history"]])
            st.dataframe(df, use_container_width=True, hide_index=True)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Descargar CSV", csv, "historial_wolfex.csv", "text/csv",
                               use_container_width=True)
