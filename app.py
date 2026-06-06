import streamlit as st
import numpy as np
import sympy as sp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

st.set_page_config(
    page_title="WolfeX",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Ocultar header y footer de streamlit */
#MainMenu, footer, header { visibility: hidden; }

/* Fondo general */
.stApp { background: #f8fafc; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0f172a !important;
    border-right: none;
    box-shadow: 4px 0 24px rgba(0,0,0,0.15);
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stNumberInput label,
section[data-testid="stSidebar"] .stTextInput label {
    color: #94a3b8 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] select {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    color: #f1f5f9 !important;
    border-radius: 8px !important;
}
section[data-testid="stSidebar"] .stButton button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.65rem !important;
    letter-spacing: 0.02em;
    box-shadow: 0 4px 15px rgba(99,102,241,0.4);
    transition: all 0.2s;
}

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 40%, #a855f7 100%);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 10px 40px rgba(99,102,241,0.3);
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%; right: -20%;
    width: 400px; height: 400px;
    background: rgba(255,255,255,0.05);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    color: white;
    margin: 0;
    line-height: 1.1;
    letter-spacing: -1px;
}
.hero-title span { color: #fde68a; }
.hero-sub {
    font-size: 1.05rem;
    color: rgba(255,255,255,0.8);
    margin-top: 0.5rem;
    font-weight: 400;
}
.hero-tags { margin-top: 1.2rem; display: flex; gap: 8px; flex-wrap: wrap; }
.tag {
    background: rgba(255,255,255,0.15);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.25);
    color: white;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 500;
}

/* Tarjetas de métrica */
.metric-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin: 1.5rem 0; }
.metric-card {
    background: white;
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    border: 1px solid #e2e8f0;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}
.metric-card.purple::before { background: linear-gradient(90deg, #6366f1, #8b5cf6); }
.metric-card.green::before  { background: linear-gradient(90deg, #10b981, #34d399); }
.metric-card.blue::before   { background: linear-gradient(90deg, #3b82f6, #60a5fa); }
.metric-label {
    font-size: 0.72rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 500;
    margin-bottom: 0.5rem;
}
.metric-val {
    font-size: 1.6rem;
    font-weight: 700;
    color: #0f172a;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1.2;
}
.metric-val.purple { color: #6366f1; }
.metric-val.green  { color: #10b981; }
.metric-val.blue   { color: #3b82f6; }

/* Info pills */
.info-strip {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    margin: 1rem 0;
    display: flex;
    flex-direction: column;
    gap: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.info-item {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 0.88rem;
    color: #475569;
}
.info-item .lbl {
    font-weight: 600;
    color: #1e293b;
    min-width: 160px;
}
.info-item code {
    background: #f1f5f9;
    color: #6366f1;
    padding: 2px 8px;
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.83rem;
}

/* Sección títulos */
.section-title {
    font-size: 1.15rem;
    font-weight: 650;
    color: #0f172a;
    margin: 1.8rem 0 0.8rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #e2e8f0;
    margin-left: 8px;
}

/* Empty state */
.empty-state {
    background: white;
    border: 2px dashed #e2e8f0;
    border-radius: 20px;
    padding: 4rem 2rem;
    text-align: center;
    margin-top: 1rem;
}
.empty-icon { font-size: 3.5rem; margin-bottom: 1rem; }
.empty-title { font-size: 1.3rem; font-weight: 600; color: #1e293b; }
.empty-sub { color: #94a3b8; margin-top: 0.4rem; font-size: 0.95rem; }

/* Success / warning banner */
.banner {
    border-radius: 12px;
    padding: 0.9rem 1.4rem;
    margin: 1rem 0;
    font-weight: 500;
    font-size: 0.92rem;
    display: flex;
    align-items: center;
    gap: 10px;
}
.banner.ok  { background: #ecfdf5; color: #065f46; border: 1px solid #a7f3d0; }
.banner.warn { background: #fffbeb; color: #92400e; border: 1px solid #fde68a; }

/* Sidebar logo */
.sb-logo {
    font-size: 1.5rem;
    font-weight: 700;
    color: white;
    letter-spacing: -0.5px;
    padding: 0.5rem 0 1.5rem;
    border-bottom: 1px solid #1e293b;
    margin-bottom: 1.2rem;
}
.sb-logo span { color: #a78bfa; }
.sb-section {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #475569;
    margin: 1.2rem 0 0.5rem;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# LÓGICA MATEMÁTICA
# ──────────────────────────────────────────────
def parse_vector(text, n):
    try:
        vals = [float(v.strip()) for v in text.replace(";", ",").split(",") if v.strip()]
        if len(vals) != n:
            raise ValueError(f"Se necesitan exactamente {n} valores.")
        return np.array(vals, dtype=float)
    except Exception as e:
        raise ValueError(f"Punto de partida inválido: {e}")


def build_functions(expr_text, n):
    syms = sp.symbols(f"x1:{n+1}")
    ld = {f"x{i+1}": syms[i] for i in range(n)}
    ld.update({"sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp,
               "log": sp.log, "ln": sp.log, "sqrt": sp.sqrt, "pi": sp.pi,
               "e": sp.E, "E": sp.E, "abs": sp.Abs, "Abs": sp.Abs,
               "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
               "atan": sp.atan, "asin": sp.asin, "acos": sp.acos})
    try:
        expr = sp.sympify(expr_text.replace("^", "**"), locals=ld)
    except Exception as e:
        raise ValueError(f"No se pudo interpretar la función: {e}")
    unk = expr.free_symbols - set(syms)
    if unk:
        raise ValueError(f"Variables no permitidas: {', '.join(str(s) for s in unk)}. Usa x1..x{n}.")
    try:
        ge = [sp.diff(expr, v) for v in syms]
        he = sp.hessian(expr, syms)
        fl = sp.lambdify((syms,), expr, "numpy")
        gl = sp.lambdify((syms,), ge, "numpy")
        hl = sp.lambdify((syms,), he, "numpy")
    except Exception as e:
        raise ValueError(f"No se pudo derivar: {e}")

    def f(x):
        v = float(np.asarray(fl(tuple(x)), dtype=float))
        if not np.isfinite(v): raise FloatingPointError("f no finita")
        return v
    def g(x):
        gv = np.array(gl(tuple(x)), dtype=float).reshape(-1)
        if not np.all(np.isfinite(gv)): raise FloatingPointError("grad no finito")
        return gv
    def h(x):
        return np.array(hl(tuple(x)), dtype=float)
    return expr, f, g, h


def wolfe_ls(f, g, x, d, a0, c1, c2, rho):
    fx = f(x); gx = g(x); dp = float(np.dot(gx, d))
    if dp >= 0: d = -gx; dp = float(np.dot(gx, d))
    a = a0; ba = None; bf = fx
    for _ in range(60):
        xn = x + a * d
        try: fn = f(xn); gn = g(xn)
        except Exception: a *= rho; continue
        if np.isfinite(fn) and fn < bf: bf = fn; ba = a
        if fn <= fx + c1*a*dp and abs(float(np.dot(gn,d))) <= c2*abs(dp):
            return a, True
        a *= rho
    return (ba, False) if ba else (a, False)


def hess_pd(H):
    H = 0.5*(H+H.T)
    try: me = float(np.min(np.linalg.eigvalsh(H)))
    except: me = -1.0
    if me > 1e-8: return H
    return H + (max(0.,-me)+1e-3)*np.eye(H.shape[0])


def run_opt(method, f, g, h, x0, max_iter, tol, a0, c1, c2, rho):
    x = x0.copy().astype(float)
    hist = []; dp = None; gp = None
    reason = "Máximo de iteraciones alcanzado"
    for k in range(max_iter+1):
        try: fx=f(x); gx=g(x)
        except: reason="Dominio inválido. Detenido."; break
        err = float(np.linalg.norm(gx))
        hist.append({"iteración":k,"f(x)":fx,"error ‖grad‖":err,"x":x.copy()})
        if err < tol: reason="Convergencia: ‖∇f‖ < tolerancia"; break
        if k == max_iter: break
        if method == "Gradiente":
            d = -gx
        elif method == "Gradiente conjugado":
            if k==0 or dp is None: d=-gx
            else:
                beta = max(0., float(np.dot(gx,gx-gp)/max(np.dot(gp,gp),1e-12)))
                d = -gx+beta*dp
                if float(np.dot(gx,d))>=0: d=-gx
        else:
            H = hess_pd(h(x))
            try: d=-np.linalg.solve(H,gx)
            except: d=-np.linalg.pinv(H).dot(gx)
            if float(np.dot(gx,d))>=0: d=-gx
        a,_ = wolfe_ls(f,g,x,d,a0,c1,c2,rho)
        xn = x+a*d; gp=gx.copy(); dp=d.copy(); x=xn
        if not np.all(np.isfinite(x)): reason="Divergencia. Detenido."; break
    if not hist: raise ValueError("No se pudo evaluar en el punto inicial.")
    try: ff=f(x)
    except: ff=hist[-1]["f(x)"]; x=hist[-1]["x"]
    return x, ff, len(hist)-1, hist, reason


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sb-logo">⚡ Wolfe<span>X</span></div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-section">Función</div>', unsafe_allow_html=True)
    n = st.number_input("Variables", min_value=1, max_value=20, value=2, step=1)
    function_text = st.text_input("f(x)", value="x1**2 + x2**2",
        help="Usa x1..xn, sin, cos, exp, log, sqrt, ^ o **")
    x0_text = st.text_input("Punto inicial x₀", value="3, 4")

    st.markdown('<div class="sb-section">Método</div>', unsafe_allow_html=True)
    method = st.selectbox("Algoritmo", ["Gradiente", "Gradiente conjugado", "Newton"])
    max_iter = st.number_input("Máx. iteraciones", 1, 100000, 100, 10)
    tol = st.number_input("Tolerancia ε", 1e-15, 1.0, 1e-6, format="%.1e")

    st.markdown('<div class="sb-section">Wolfe</div>', unsafe_allow_html=True)
    alpha0 = st.number_input("α inicial", 1e-12, 100.0, 1.0, format="%.4f")
    c1 = st.number_input("c₁ (Armijo)", 1e-12, 0.5, 1e-4, format="%.1e")
    c2 = st.number_input("c₂ (curvatura)", 0.01, 0.99, 0.9, format="%.2f")
    rho = st.number_input("ρ reducción", 0.01, 0.99, 0.5, format="%.2f")

    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("⚡ Ejecutar", type="primary", use_container_width=True)


# ──────────────────────────────────────────────
# HERO
# ──────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-title">Métodos de <span>Optimización</span></div>
  <div class="hero-sub">Gradiente · Gradiente Conjugado · Newton con condiciones de Wolfe fuerte</div>
  <div class="hero-tags">
    <span class="tag">📐 Primer Semestre 2026</span>
    <span class="tag">🔬 Búsqueda de línea exacta</span>
    <span class="tag">⚡ SymPy + NumPy</span>
    <span class="tag">📊 Convergencia visual</span>
  </div>
</div>
""", unsafe_allow_html=True)

# Ejemplos colapsables
with st.expander("📋 Ver ejemplos de funciones"):
    e1, e2, e3 = st.columns(3)
    with e1:
        st.code("x1**2 + x2**2", language=None)
        st.caption("Esférica · mínimo (0, 0)")
    with e2:
        st.code("(x1-1)**2 + (x2+2)**2", language=None)
        st.caption("Desplazada · mínimo (1, -2)")
    with e3:
        st.code("100*(x2-x1**2)**2 + (1-x1)**2", language=None)
        st.caption("Rosenbrock · mínimo (1, 1)")

# ──────────────────────────────────────────────
# RESULTADOS
# ──────────────────────────────────────────────
if not run:
    st.markdown("""
    <div class="empty-state">
      <div class="empty-icon">🎯</div>
      <div class="empty-title">Listo para optimizar</div>
      <div class="empty-sub">Configura los parámetros en el panel izquierdo y presiona <strong>Ejecutar</strong></div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

try:
    if not (0 < c1 < c2 < 1):
        st.error("Se requiere 0 < c₁ < c₂ < 1 para las condiciones de Wolfe.")
        st.stop()

    with st.spinner("⚡ Optimizando..."):
        expr, f, g, h = build_functions(function_text, int(n))
        x0 = parse_vector(x0_text, int(n))
        xmin, fmin, iters, history, reason = run_opt(
            method, f, g, h, x0, int(max_iter), tol, alpha0, c1, c2, rho)

    convergio = "Convergencia" in reason
    bclass = "ok" if convergio else "warn"
    bicon = "✅" if convergio else "⚠️"
    st.markdown(f'<div class="banner {bclass}">{bicon} {reason}</div>', unsafe_allow_html=True)

    # Métricas
    xstr = np.array2string(xmin, precision=5, suppress_small=True)
    st.markdown(f"""
    <div class="metric-grid">
      <div class="metric-card purple">
        <div class="metric-label">Punto mínimo x*</div>
        <div class="metric-val purple">{xstr}</div>
      </div>
      <div class="metric-card green">
        <div class="metric-label">Valor f(x*)</div>
        <div class="metric-val green">{fmin:.6g}</div>
      </div>
      <div class="metric-card blue">
        <div class="metric-label">Iteraciones</div>
        <div class="metric-val blue">{iters}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    fe = history[-1]["error ‖grad‖"]
    st.markdown(f"""
    <div class="info-strip">
      <div class="info-item"><span class="lbl">Error final ‖∇f‖</span> <code>{fe:.4e}</code></div>
      <div class="info-item"><span class="lbl">Criterio de parada</span> <code>{reason}</code></div>
      <div class="info-item"><span class="lbl">Función interpretada</span> <code>{sp.sstr(expr)}</code></div>
      <div class="info-item"><span class="lbl">Método usado</span> <code>{method}</code></div>
    </div>
    """, unsafe_allow_html=True)

    # Tabla
    st.markdown('<div class="section-title">📊 Historial de iteraciones</div>', unsafe_allow_html=True)
    df = pd.DataFrame([{
        "it.": h2["iteración"],
        "f(x)": round(h2["f(x)"], 8),
        "‖∇f‖": round(h2["error ‖grad‖"], 8),
        "x": np.array2string(h2["x"], precision=4, suppress_small=True)
    } for h2 in history])
    st.dataframe(df, use_container_width=True, hide_index=True)

    csv = pd.DataFrame(history).to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar CSV", csv, "historial.csv", "text/csv")

    # Gráfico
    st.markdown('<div class="section-title">📈 Convergencia</div>', unsafe_allow_html=True)
    errs = np.array([h2["error ‖grad‖"] for h2 in history])
    its  = np.array([h2["iteración"] for h2 in history])
    pos  = errs[errs > 0]
    log  = len(pos) >= 2 and errs.max()/pos.min() > 100

    fig, ax = plt.subplots(figsize=(11, 4.5))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8fafc")

    color_main  = "#6366f1"
    color_fill  = "#ede9fe"

    if log:
        ax.semilogy(its, np.maximum(errs, 1e-16),
                    color=color_main, lw=2.5, marker="o", ms=5,
                    markerfacecolor="white", markeredgecolor=color_main, markeredgewidth=2)
        ax.fill_between(its, 1e-16, np.maximum(errs, 1e-16), alpha=0.08, color=color_main)
        ax.set_ylabel("‖∇f‖  (escala log)", color="#475569", fontsize=11)
    else:
        ax.plot(its, errs, color=color_main, lw=2.5, marker="o", ms=5,
                markerfacecolor="white", markeredgecolor=color_main, markeredgewidth=2)
        ax.fill_between(its, 0, errs, alpha=0.08, color=color_main)
        ax.set_ylabel("‖∇f‖", color="#475569", fontsize=11)

    ax.set_xlabel("Iteración", color="#475569", fontsize=11)
    ax.set_title(f"Convergencia — {method}", color="#0f172a", fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(colors="#94a3b8", labelsize=9)
    for sp2 in ax.spines.values(): sp2.set_color("#e2e8f0")
    ax.grid(True, alpha=0.5, color="#e2e8f0", linestyle="--")
    ax.set_xlim(left=0)
    fig.tight_layout()
    st.pyplot(fig)

except Exception as ex:
    st.error(f"❌ {ex}")
