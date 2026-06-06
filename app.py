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

.topbar {
    background: rgba(255,255,255,0.03);
    border-bottom: 1px solid rgba(255,255,255,0.06);
    padding: 1rem 2.5rem;
    display: flex; align-items: center; justify-content: space-between;
}
.topbar-logo {
    font-size: 1.4rem; font-weight: 800; letter-spacing: -0.5px;
    background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.nav-chip {
    background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.08);
    color: #94a3b8; border-radius: 8px; padding: 6px 14px;
    font-size: 0.8rem; font-weight: 500; margin-left: 6px;
}
.nav-chip.active {
    background: rgba(139,92,246,0.2); border-color: rgba(139,92,246,0.4); color: #a78bfa;
}

/* Tutorial steps */
.tut-step {
    background: #0d1117; border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px; padding: 1.2rem 1.4rem; margin-bottom: 10px;
    display: flex; gap: 14px; align-items: flex-start;
}
.tut-num {
    background: linear-gradient(135deg,#7c3aed,#6366f1);
    color: white; border-radius: 50%; width: 32px; height: 32px;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 0.9rem; flex-shrink: 0;
}
.tut-title { font-weight: 600; color: #e2e8f0; font-size: 0.95rem; margin-bottom: 4px; }
.tut-desc  { color: #64748b; font-size: 0.83rem; line-height: 1.6; }
.tut-code  {
    display: inline-block; background: rgba(167,139,250,0.1);
    color: #a78bfa; border: 1px solid rgba(167,139,250,0.2);
    border-radius: 6px; padding: 2px 8px;
    font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;
}

/* Explicación resultado */
.explain-box {
    border-radius: 14px; padding: 1.3rem 1.5rem; margin: 8px 0;
    border-left: 4px solid;
}
.explain-box.ok   { background: rgba(52,211,153,0.07); border-color: #34d399; }
.explain-box.warn { background: rgba(251,191,36,0.07);  border-color: #fbbf24; }
.explain-box.info { background: rgba(96,165,250,0.07);  border-color: #60a5fa; }
.explain-title { font-weight: 700; font-size: 1rem; margin-bottom: 6px; }
.explain-box.ok   .explain-title { color: #34d399; }
.explain-box.warn .explain-title { color: #fbbf24; }
.explain-box.info .explain-title { color: #60a5fa; }
.explain-body { color: #94a3b8; font-size: 0.88rem; line-height: 1.7; }
.explain-body strong { color: #cbd5e1; }

/* Streamlit overrides */
section[data-testid="stSidebar"] { display: none; }
.stTextInput input, .stNumberInput input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 8px !important; color: #e2e8f0 !important;
    font-family: 'JetBrains Mono', monospace !important;
}
.stTextInput label, .stNumberInput label, .stSelectbox label {
    color: #64748b !important; font-size: 0.72rem !important;
    text-transform: uppercase !important; letter-spacing: 0.08em !important;
}
div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# MATEMÁTICA
# ──────────────────────────────────────────────
def fmt_expr(expr_str):
    """Convierte ** a ^ para mostrar al usuario."""
    return expr_str.replace("**", "^")

def parse_vector(text, n):
    try:
        vals = [float(v.strip()) for v in text.replace(";",",").split(",") if v.strip()]
        if len(vals) != n: raise ValueError(f"Se necesitan {n} valores.")
        return np.array(vals, dtype=float)
    except Exception as e:
        raise ValueError(f"Punto inválido: {e}")

def build_functions(expr_text, n):
    # Aceptar ^ como potencia
    expr_text = expr_text.replace("^","**")
    syms = sp.symbols(f"x1:{n+1}")
    ld = {f"x{i+1}": syms[i] for i in range(n)}
    ld.update({"sin":sp.sin,"cos":sp.cos,"tan":sp.tan,"exp":sp.exp,"log":sp.log,
               "ln":sp.log,"sqrt":sp.sqrt,"pi":sp.pi,"e":sp.E,"E":sp.E,
               "abs":sp.Abs,"Abs":sp.Abs,"sinh":sp.sinh,"cosh":sp.cosh,
               "tanh":sp.tanh,"atan":sp.atan,"asin":sp.asin,"acos":sp.acos})
    try:
        expr = sp.sympify(expr_text, locals=ld)
    except Exception as e:
        raise ValueError(f"No se pudo interpretar la función: {e}")
    unk = expr.free_symbols - set(syms)
    if unk: raise ValueError(f"Variables no permitidas: {', '.join(str(s) for s in unk)}. Usa x1..x{n}.")
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

def generar_explicacion(res):
    """Genera explicación didáctica del resultado en lenguaje simple."""
    fmin   = res["fmin"]
    iters  = res["iters"]
    reason = res["reason"]
    method = res["method"]
    fe     = res["history"][-1]["‖∇f‖"]
    xmin   = res["xmin"]
    xstr   = np.array2string(xmin, precision=4, suppress_small=True)
    convergio = "Convergencia" in reason

    # ¿Es mínimo?
    if convergio:
        exp_minimo = f"""
<div class="explain-box ok">
  <div class="explain-title">✅ Se encontró un mínimo local</div>
  <div class="explain-body">
    El algoritmo llegó al punto <strong>{xstr}</strong> donde el valor de la función es <strong>{fmin:.6g}</strong>.
    Sabemos que es un <strong>mínimo</strong> porque el gradiente (la "pendiente" de la función en ese punto)
    es prácticamente cero — su norma es <strong>{fe:.2e}</strong>, menor que la tolerancia pedida.
    En términos simples: la función "deja de bajar" ahí, lo que matemáticamente define un punto mínimo.
  </div>
</div>"""
    else:
        exp_minimo = f"""
<div class="explain-box warn">
  <div class="explain-title">⚠️ No se confirmó convergencia al mínimo</div>
  <div class="explain-body">
    El método se detuvo pero <strong>no porque encontró el mínimo</strong>, sino porque se alcanzó
    el límite de iteraciones o hubo un problema numérico. El error final es <strong>{fe:.2e}</strong>,
    que aún puede ser grande. El punto encontrado <strong>{xstr}</strong> es el mejor que se alcanzó,
    pero puede no ser el mínimo real. Prueba aumentar las iteraciones máximas o cambiar el punto de partida.
  </div>
</div>"""

    # Iteraciones
    method_speeds = {
        "Newton": ("muy rápido — generalmente converge en pocas iteraciones porque usa información de la curvatura (Hessiano) de la función.",
                   "Si necesitó muchas iteraciones, la función probablemente es muy no-lineal o el punto de partida estaba lejos del mínimo."),
        "Gradiente conjugado": ("moderadamente rápido — mejora el gradiente puro usando direcciones conjugadas que no interfieren entre sí.",
                                "Más iteraciones que Newton es normal; es más eficiente que el gradiente simple."),
        "Gradiente": ("el más lento de los tres — avanza siempre en la dirección de mayor descenso pero sin considerar la curvatura.",
                      "Muchas iteraciones son normales, especialmente en funciones con 'valles estrechos' como Rosenbrock."),
    }
    spd, spd2 = method_speeds.get(method, ("",""))

    exp_iters = f"""
<div class="explain-box info">
  <div class="explain-title">🔁 ¿Por qué {iters} iteraciones?</div>
  <div class="explain-body">
    El método <strong>{method}</strong> es {spd}<br><br>
    {spd2}<br><br>
    Cada iteración calcula una nueva dirección de descenso y luego usa la <strong>búsqueda de línea de Wolfe</strong>
    para encontrar el tamaño de paso óptimo (ni muy grande que salte el mínimo, ni muy pequeño que avance lentísimo).
    El proceso se repite hasta que el gradiente sea suficientemente pequeño o se agoten las iteraciones.
  </div>
</div>"""

    # ¿Mínimo global o local?
    exp_tipo = f"""
<div class="explain-box info">
  <div class="explain-title">📌 ¿Mínimo local o global?</div>
  <div class="explain-body">
    Estos métodos garantizan encontrar un <strong>mínimo local</strong> — el punto más bajo en la
    "vecindad" del punto de partida. Para funciones convexas (como x1²+x2²), el mínimo local
    <em>es</em> el global. Para funciones no convexas (como Rosenbrock), puede haber múltiples mínimos
    locales y el resultado depende del punto de partida. Si sospechas que no encontraste el mínimo global,
    prueba con diferentes puntos de partida.
  </div>
</div>"""

    return exp_minimo + exp_iters + exp_tipo


# ──────────────────────────────────────────────
# ESTADO
# ──────────────────────────────────────────────
for k, v in [("method","Newton"),("func","x1^2 + x2^2"),
             ("x0","3, 4"),("resultado",None),("tutorial",False)]:
    if k not in st.session_state:
        st.session_state[k] = v


# ──────────────────────────────────────────────
# TOPBAR
# ──────────────────────────────────────────────
st.markdown("""
<div class="topbar">
  <div class="topbar-logo">⚡ WolfeX</div>
  <div>
    <span class="nav-chip active">Optimizador</span>
    <span class="nav-chip">Wolfe Fuerte</span>
    <span class="nav-chip">Semestre 2026</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# TUTORIAL INTERACTIVO (expander destacado)
# ──────────────────────────────────────────────
with st.expander("🎓 Tutorial interactivo — ¿Cómo usar WolfeX? (haz clic aquí si es tu primera vez)", expanded=False):
    st.markdown("""
<div style="padding:0.5rem 0 1rem;">
  <div style="font-size:1.1rem;font-weight:700;color:#e2e8f0;margin-bottom:1rem;">
    Bienvenido/a 👋 Aprende a usar WolfeX en 5 pasos simples
  </div>

  <div class="tut-step">
    <div class="tut-num">1</div>
    <div>
      <div class="tut-title">Escribe tu función matemática</div>
      <div class="tut-desc">
        En el campo <em>"Función f(x)"</em>, escribe la función que quieres minimizar.<br>
        Usa <span class="tut-code">x1</span>, <span class="tut-code">x2</span>, etc. para las variables
        y <span class="tut-code">^</span> para elevar a una potencia.<br>
        Ejemplo: <span class="tut-code">x1^2 + x2^2</span> significa x₁² + x₂²<br>
        También puedes usar los botones de ejemplos rápidos: <strong>Esférica</strong>, <strong>Desplazada</strong> o <strong>Rosenbrock</strong>.
      </div>
    </div>
  </div>

  <div class="tut-step">
    <div class="tut-num">2</div>
    <div>
      <div class="tut-title">Elige el número de variables</div>
      <div class="tut-desc">
        Si tu función tiene x1 y x2, pon <span class="tut-code">2</span>. Si tiene x1, x2 y x3, pon <span class="tut-code">3</span>, y así sucesivamente.
        La mayoría de los ejemplos usan 2 variables.
      </div>
    </div>
  </div>

  <div class="tut-step">
    <div class="tut-num">3</div>
    <div>
      <div class="tut-title">Ingresa el punto de partida</div>
      <div class="tut-desc">
        El algoritmo necesita un punto inicial desde donde empezar a buscar el mínimo.
        Escríbelo como números separados por coma. Ejemplo: <span class="tut-code">3, 4</span> significa x1=3, x2=4.<br>
        Si tu función tiene 3 variables, escribe 3 números: <span class="tut-code">1, 2, 3</span>
      </div>
    </div>
  </div>

  <div class="tut-step">
    <div class="tut-num">4</div>
    <div>
      <div class="tut-title">Selecciona el método</div>
      <div class="tut-desc">
        Hay 3 métodos disponibles. Si no sabes cuál elegir, usa <strong>Newton</strong> — es el más rápido y preciso para la mayoría de funciones.<br>
        • <strong>Gradiente</strong>: el más simple, funciona siempre pero puede ser lento.<br>
        • <strong>Gradiente Conjugado</strong>: intermedio, más rápido que el gradiente puro.<br>
        • <strong>Newton</strong>: el más rápido, usa la curvatura de la función para dar mejores saltos.
      </div>
    </div>
  </div>

  <div class="tut-step">
    <div class="tut-num">5</div>
    <div>
      <div class="tut-title">Presiona ⚡ Ejecutar y lee los resultados</div>
      <div class="tut-desc">
        Haz clic en el botón <strong>Ejecutar optimización</strong> y la app te mostrará:<br>
        • El punto donde está el mínimo (las coordenadas x1, x2, ...)<br>
        • El valor más bajo que alcanzó la función<br>
        • Cuántas iteraciones (pasos) necesitó<br>
        • Una explicación en lenguaje simple de qué significa cada resultado<br>
        • Un gráfico mostrando cómo fue bajando el error paso a paso
      </div>
    </div>
  </div>

  <div style="background:rgba(52,211,153,0.07);border:1px solid rgba(52,211,153,0.2);border-radius:10px;padding:1rem 1.2rem;margin-top:0.5rem;">
    <div style="color:#34d399;font-weight:600;margin-bottom:4px;">💡 Consejo rápido</div>
    <div style="color:#64748b;font-size:0.85rem;">
      Si nunca has usado esta app, prueba primero haciendo clic en <strong>"Esférica"</strong>
      en los ejemplos rápidos y luego en <strong>Ejecutar</strong>. Verás un resultado simple con explicación detallada.
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# LAYOUT PRINCIPAL
# ──────────────────────────────────────────────
left, right = st.columns([1.1, 2.4])

with left:
    st.markdown("""
    <div style="font-size:0.68rem;font-weight:600;color:#475569;
         text-transform:uppercase;letter-spacing:0.12em;margin-bottom:0.8rem;">
      ⚙️ Configuración
    </div>""", unsafe_allow_html=True)

    # Función — con notación ^
    func_raw = st.text_input(
        "Función f(x)  — usa ^ para potencias",
        value=st.session_state.func,
        help="Ejemplo: x1^2 + x2^2 · También puedes usar sin, cos, exp, log, sqrt"
    )
    st.session_state.func = func_raw
    # Preview con notación ^
    preview = func_raw.replace("**","^")
    st.markdown(f"<div style='font-family:JetBrains Mono,monospace;font-size:0.8rem;color:#a78bfa;margin:-8px 0 8px;'>f(x) = {preview}</div>", unsafe_allow_html=True)

    # Botones ejemplos rápidos
    st.markdown("<div style='font-size:0.7rem;color:#475569;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px;'>⚡ Ejemplos rápidos</div>", unsafe_allow_html=True)
    ec1, ec2, ec3 = st.columns(3)
    if ec1.button("🔵 Esférica", use_container_width=True, help="x1² + x2² · mínimo en (0,0)"):
        st.session_state.func = "x1^2 + x2^2"
        st.session_state.x0 = "3, 4"
        st.rerun()
    if ec2.button("🟣 Desplazada", use_container_width=True, help="(x1-1)² + (x2+2)² · mínimo en (1,-2)"):
        st.session_state.func = "(x1-1)^2 + (x2+2)^2"
        st.session_state.x0 = "0, 0"
        st.rerun()
    if ec3.button("🟠 Rosenbrock", use_container_width=True, help="Función clásica difícil · mínimo en (1,1)"):
        st.session_state.func = "100*(x2-x1^2)^2 + (1-x1)^2"
        st.session_state.x0 = "-1.2, 1"
        st.rerun()

    st.markdown("---")

    # Método con botones destacados
    st.markdown("<div style='font-size:0.7rem;color:#475569;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;'>🧮 Método de optimización</div>", unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    metodos = [("Gradiente","▽","#a78bfa","m1",m1),
               ("Gradiente conjugado","⇉","#60a5fa","m2",m2),
               ("Newton","⊕","#34d399","m3",m3)]
    for lbl, icon, clr, key, col in metodos:
        sel = st.session_state.method == lbl
        with col:
            if st.button(
                f"{icon} {'✓ ' if sel else ''}{lbl.split()[0]}",
                key=key,
                use_container_width=True,
                type="primary" if sel else "secondary",
                help=f"Seleccionar método {lbl}"
            ):
                st.session_state.method = lbl
                st.rerun()

    met_desc = {
        "Gradiente": "▽ Gradiente — sigue la pendiente más pronunciada. Simple y seguro.",
        "Gradiente conjugado": "⇉ Conjugado — combina dirección actual con la anterior. Más rápido.",
        "Newton": "⊕ Newton — usa la curvatura de la función. El más veloz y preciso.",
    }
    st.markdown(f"<div style='background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:8px;padding:8px 12px;font-size:0.8rem;color:#64748b;margin-top:6px;'>{met_desc[st.session_state.method]}</div>", unsafe_allow_html=True)

    st.markdown("---")

    n      = st.number_input("Número de variables (x1, x2, ...)", 1, 20, 2, 1)
    x0_txt = st.text_input("Punto de partida (ej: 3, 4)", value=st.session_state.x0)
    st.session_state.x0 = x0_txt
    max_it = st.number_input("Máx. iteraciones", 1, 100000, 100, 10)
    tol    = st.number_input("Tolerancia ε", 1e-15, 1.0, 1e-6, format="%.1e",
                              help="Qué tan pequeño debe ser el gradiente para declarar convergencia")

    with st.expander("🔧 Parámetros avanzados de Wolfe"):
        st.caption("Modifica solo si sabes lo que haces. Los valores por defecto funcionan bien.")
        alpha0 = st.number_input("α inicial", 1e-12, 100.0, 1.0, format="%.4f")
        c1     = st.number_input("c₁ Armijo",  1e-12, 0.5,  1e-4, format="%.1e")
        c2     = st.number_input("c₂ curvatura", 0.01, 0.99, 0.9, format="%.2f")
        rho    = st.number_input("ρ reducción α", 0.01, 0.99, 0.5, format="%.2f")
    alpha0 = locals().get("alpha0", 1.0)
    c1     = locals().get("c1", 1e-4)
    c2     = locals().get("c2", 0.9)
    rho    = locals().get("rho", 0.5)

    st.markdown("---")
    run = st.button("⚡  Ejecutar optimización", type="primary", use_container_width=True)

    if run:
        try:
            if not (0 < c1 < c2 < 1):
                st.error("Se requiere 0 < c₁ < c₂ < 1")
            else:
                with st.spinner("⚡ Calculando..."):
                    expr, f, g, h = build_functions(st.session_state.func, int(n))
                    x0v = parse_vector(st.session_state.x0, int(n))
                    xmin, fmin, iters, history, reason = run_opt(
                        st.session_state.method, f, g, h, x0v,
                        int(max_it), tol, alpha0, c1, c2, rho)
                st.session_state.resultado = {
                    "xmin": xmin, "fmin": fmin, "iters": iters,
                    "history": history, "reason": reason,
                    "expr": fmt_expr(sp.sstr(expr)),
                    "method": st.session_state.method
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
        st.markdown("""
        <div style="padding:4rem 2rem;text-align:center;">
          <div style="font-size:3.5rem;font-weight:800;letter-spacing:-2px;
               background:linear-gradient(135deg,#fff 0%,#a78bfa 50%,#60a5fa 100%);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;
               margin-bottom:1rem;">
            Optimización<br>Numérica
          </div>
          <div style="color:#334155;font-size:0.95rem;max-width:420px;margin:0 auto 2.5rem;line-height:1.8;">
            Encuentra el punto mínimo de cualquier función matemática usando tres algoritmos clásicos,
            todos con búsqueda de línea que cumple las condiciones de Wolfe.
          </div>
          <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;max-width:500px;margin:0 auto 2rem;">
            <div style="background:#0d1117;border:1px solid rgba(255,255,255,0.06);border-radius:14px;padding:1.3rem 0.8rem;text-align:center;">
              <div style="font-size:2rem">▽</div>
              <div style="font-weight:600;color:#c4b5fd;margin:6px 0 2px;font-size:0.85rem;">Gradiente</div>
              <div style="font-size:0.72rem;color:#334155;">Descenso clásico</div>
            </div>
            <div style="background:#0d1117;border:1px solid rgba(255,255,255,0.06);border-radius:14px;padding:1.3rem 0.8rem;text-align:center;">
              <div style="font-size:2rem">⇉</div>
              <div style="font-weight:600;color:#60a5fa;margin:6px 0 2px;font-size:0.85rem;">Conjugado</div>
              <div style="font-size:0.72rem;color:#334155;">Polak-Ribière</div>
            </div>
            <div style="background:#0d1117;border:1px solid rgba(255,255,255,0.06);border-radius:14px;padding:1.3rem 0.8rem;text-align:center;">
              <div style="font-size:2rem">⊕</div>
              <div style="font-weight:600;color:#34d399;margin:6px 0 2px;font-size:0.85rem;">Newton</div>
              <div style="font-size:0.72rem;color:#334155;">Hessiano reg.</div>
            </div>
          </div>
          <div style="color:#1e293b;font-size:0.85rem;">
            ← Configura los parámetros y presiona <strong style="color:#a78bfa">⚡ Ejecutar</strong>
          </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        convergio = "Convergencia" in res["reason"]
        bstyle = "background:rgba(52,211,153,.1);color:#34d399;border:1px solid rgba(52,211,153,.25);" if convergio \
                 else "background:rgba(251,191,36,.1);color:#fbbf24;border:1px solid rgba(251,191,36,.25);"
        bicon = "✅" if convergio else "⚠️"

        st.markdown(f"""
        <div style="border-radius:10px;padding:10px 16px;font-size:0.88rem;
             font-weight:600;margin-bottom:1rem;{bstyle}">
          {bicon} &nbsp;{res['reason']}
        </div>""", unsafe_allow_html=True)

        # KPIs
        xstr = np.array2string(res["xmin"], precision=5, suppress_small=True)
        c1k, c2k, c3k = st.columns(3)
        c1k.metric("📍 Punto mínimo x*", xstr)
        c2k.metric("🎯 Valor f(x*)", f"{res['fmin']:.6g}")
        c3k.metric("🔁 Iteraciones", res["iters"])

        fe = res["history"][-1]["‖∇f‖"]
        ia, ib = st.columns(2)
        ia.info(f"**Error final ‖∇f‖:** `{fe:.4e}`")
        ib.info(f"**Función:** `{res['expr']}`")

        # ── EXPLICACIÓN AUTOMÁTICA ──
        st.markdown("""
        <div style="font-size:0.72rem;font-weight:600;color:#475569;
             text-transform:uppercase;letter-spacing:0.12em;margin:1.4rem 0 0.6rem;">
          🧠 Explicación del resultado
        </div>""", unsafe_allow_html=True)
        st.markdown(generar_explicacion(res), unsafe_allow_html=True)

        # ── TABS: Gráfico + Tabla ──
        st.markdown("<div style='margin-top:1.2rem'></div>", unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["📈 Gráfico de convergencia", "📊 Historial de iteraciones"])

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
            y = np.maximum(errs, 1e-16)
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
            ax.set_title(f"Convergencia — {res['method']}", color="#e2e8f0", fontsize=12, fontweight="bold")
            ax.tick_params(colors="#334155", labelsize=8)
            for s in ax.spines.values(): s.set_color("#1e293b")
            ax.grid(True, alpha=0.3, color="#1e293b", linestyle="--")
            ax.legend(fontsize=9, framealpha=0.1, labelcolor=col)
            fig.tight_layout()
            st.pyplot(fig)
            st.caption("El gráfico muestra cómo el error (norma del gradiente) disminuye con cada iteración. Cuando llega a 0 (o muy cerca), el algoritmo encontró el mínimo.")

        with tab2:
            df = pd.DataFrame([{
                "it.": h["iteración"],
                "f(x)": round(h["f(x)"],8),
                "‖∇f‖": round(h["‖∇f‖"],8),
                "x": np.array2string(h["x"],precision=4,suppress_small=True)
            } for h in res["history"]])
            st.dataframe(df, use_container_width=True, hide_index=True)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Descargar historial CSV",
                csv, "historial_wolfex.csv", "text/csv",
                use_container_width=True
            )
