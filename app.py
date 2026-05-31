import streamlit as st
import numpy as np
import sympy as sp
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Optimizador con Wolfe", layout="wide")
st.title("Proyecto Final - Métodos de Optimización")
st.caption("Gradiente, Gradiente Conjugado y Newton con condiciones de Wolfe")

# ------------------------------
# Utilidades
# ------------------------------
def parse_vector(text, n):
    try:
        vals = [float(v.strip()) for v in text.replace(";", ",").split(",") if v.strip() != ""]
        if len(vals) != n:
            raise ValueError(f"Debe ingresar exactamente {n} valores.")
        return np.array(vals, dtype=float)
    except Exception as e:
        raise ValueError(f"Punto de partida inválido: {e}")


def build_functions(expr_text, n):
    x_symbols = sp.symbols("x1:" + str(n + 1))
    local_dict = {f"x{i+1}": x_symbols[i] for i in range(n)}
    local_dict.update({
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp,
        "log": sp.log, "ln": sp.log, "sqrt": sp.sqrt, "pi": sp.pi,
        "e": sp.E, "E": sp.E, "abs": sp.Abs, "Abs": sp.Abs,
        "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
        "atan": sp.atan, "asin": sp.asin, "acos": sp.acos,
    })
    try:
        expr = sp.sympify(expr_text.replace("^", "**"), locals=local_dict)
    except Exception as e:
        raise ValueError(f"No se pudo interpretar la función objetivo: {e}")

    # Rechazar variables fuera de x1..xn (p. ej. x5 con n=2, o 'a', 'k')
    desconocidos = expr.free_symbols - set(x_symbols)
    if desconocidos:
        nombres = ", ".join(sorted(str(s) for s in desconocidos))
        raise ValueError(
            f"La función usa variables no permitidas: {nombres}. "
            f"Usa solo x1..x{n}."
        )

    try:
        grad_expr = [sp.diff(expr, v) for v in x_symbols]
        hess_expr = sp.hessian(expr, x_symbols)
        f_np = sp.lambdify((x_symbols,), expr, "numpy")
        g_np = sp.lambdify((x_symbols,), grad_expr, "numpy")
        h_np = sp.lambdify((x_symbols,), hess_expr, "numpy")
    except Exception as e:
        raise ValueError(f"No se pudo derivar la función objetivo: {e}")

    def f(x):
        val = float(np.asarray(f_np(tuple(x)), dtype=float))
        if not np.isfinite(val):
            raise FloatingPointError("f(x) no es finita")
        return val

    def g(x):
        gv = np.array(g_np(tuple(x)), dtype=float).reshape(-1)
        if not np.all(np.isfinite(gv)):
            raise FloatingPointError("gradiente no finito")
        return gv

    def h(x):
        return np.array(h_np(tuple(x)), dtype=float)

    return expr, f, g, h



def wolfe_line_search(f, g, x, d, alpha0=1.0, c1=1e-4, c2=0.9, rho=0.5, max_ls=50):
    """Búsqueda de línea con la primera y segunda condición de Wolfe (forma fuerte).
    Primera (Armijo):    f(x+a d) <= f(x) + c1*a*grad(x)^T d
    Segunda (curvatura): |grad(x+a d)^T d| <= c2*|grad(x)^T d|
    """
    alpha = alpha0
    fx = f(x)
    gx = g(x)
    phi0_der = float(np.dot(gx, d))

    if phi0_der >= 0:
        d = -gx
        phi0_der = float(np.dot(gx, d))

    best_alpha = None
    best_f = fx

    for _ in range(max_ls):
        x_new = x + alpha * d
        try:
            f_new = f(x_new)
            g_new = g(x_new)
        except Exception:
            alpha *= rho
            continue

        # Respaldo: guardamos el mejor paso que al menos reduce f
        if np.isfinite(f_new) and f_new < best_f:
            best_f = f_new
            best_alpha = alpha

        cond1 = f_new <= fx + c1 * alpha * phi0_der
        cond2 = abs(float(np.dot(g_new, d))) <= c2 * abs(phi0_der)

        if cond1 and cond2:
            return alpha, True
        alpha *= rho

    # Si nada cumple Wolfe, usamos el mejor paso que redujo f (evita estancarse)
    if best_alpha is not None:
        return best_alpha, False
    return alpha, False


def hessiano_definido_positivo(H):
    """Devuelve un Hessiano definido positivo. Si H no lo es (autovalores <= 0),
    se le suma tau*I (modificación de Levenberg). Esto permite que Newton
    funcione también con funciones no convexas sin atascarse ni divergir."""
    H = 0.5 * (H + H.T)  # simetrizar por seguridad numérica
    n = H.shape[0]
    try:
        min_eig = float(np.min(np.linalg.eigvalsh(H)))
    except np.linalg.LinAlgError:
        min_eig = -1.0
    if min_eig > 1e-8:
        return H
    tau = max(0.0, -min_eig) + 1e-3
    return H + tau * np.eye(n)


def optimize(method, f, g, h, x0, max_iter, tol, alpha0, c1, c2, rho):
    x = x0.astype(float).copy()
    history = []
    d_prev = None
    g_prev = None
    stop_reason = "Máximo de iteraciones alcanzado"

    for k in range(max_iter + 1):
        try:
            fx = f(x)
            gx = g(x)
        except (FloatingPointError, ValueError, OverflowError, ZeroDivisionError):
            stop_reason = "Evaluación no válida (dominio de la función). Detenido."
            break
        error = float(np.linalg.norm(gx))
        history.append({"iteración": k, "f(x)": fx, "error ||grad||": error, "x": x.copy()})

        if error < tol:
            stop_reason = "Convergencia alcanzada: ||gradiente|| < tolerancia"
            break

        if k == max_iter:
            break

        if method == "Gradiente":
            d = -gx

        elif method == "Gradiente conjugado":
            if k == 0 or d_prev is None or g_prev is None:
                d = -gx
            else:
                beta = max(0.0, float(np.dot(gx, gx - g_prev) / max(np.dot(g_prev, g_prev), 1e-12)))
                d = -gx + beta * d_prev
                if float(np.dot(gx, d)) >= 0:
                    d = -gx

        elif method == "Newton":
            H = hessiano_definido_positivo(h(x))
            try:
                d = -np.linalg.solve(H, gx)
            except np.linalg.LinAlgError:
                d = -np.linalg.pinv(H).dot(gx)
            if float(np.dot(gx, d)) >= 0:
                d = -gx
        else:
            raise ValueError("Método no reconocido")

        alpha, wolfe_ok = wolfe_line_search(f, g, x, d, alpha0, c1, c2, rho)
        x_new = x + alpha * d

        g_prev = gx.copy()
        d_prev = d.copy()
        x = x_new

        if not np.all(np.isfinite(x)):
            stop_reason = "El método generó valores no finitos"
            break

    if not history:
        raise ValueError(
            "No se pudo evaluar la función en el punto de partida "
            "(¿dominio inválido?). Prueba otro punto inicial."
        )
    try:
        f_final = f(x)
    except Exception:
        f_final = history[-1]["f(x)"]
        x = history[-1]["x"]
    return x, f_final, len(history) - 1, history, stop_reason


# ------------------------------
# Interfaz
# ------------------------------
with st.sidebar:
    st.header("Datos de entrada")
    n = st.number_input("Número de variables", min_value=1, max_value=5, value=2, step=1)
    method = st.selectbox("Método de optimización", ["Gradiente", "Gradiente conjugado", "Newton"])
    function_text = st.text_input("Función objetivo", value="x1**2 + x2**2")
    x0_text = st.text_input("Punto de partida", value="3, 4")
    max_iter = st.number_input("Número máximo de iteraciones", min_value=1, max_value=10000, value=100, step=10)
    tol = st.number_input("Tolerancia de convergencia", min_value=1e-12, max_value=1.0, value=1e-6, format="%.1e")

    st.subheader("Parámetros Wolfe")
    alpha0 = st.number_input("Alpha inicial", min_value=1e-12, max_value=100.0, value=1.0, format="%.4f")
    c1 = st.number_input("c1 primera condición Wolfe", min_value=1e-12, max_value=0.5, value=1e-4, format="%.1e")
    c2 = st.number_input("c2 segunda condición Wolfe", min_value=0.01, max_value=0.99, value=0.9, format="%.2f")
    rho = st.number_input("Factor de reducción alpha", min_value=0.01, max_value=0.99, value=0.5, format="%.2f")
    run = st.button("Ejecutar optimización", type="primary")

st.markdown("""
Esta aplicación permite encontrar el mínimo de una función usando tres métodos: **gradiente**,
**gradiente conjugado** y **Newton**. Todos usan búsqueda de línea con la **primera y segunda condición de Wolfe**.

Ejemplos de función:
- `x1**2 + x2**2`
- `(x1-1)**2 + (x2+2)**2`
- `100*(x2-x1**2)**2 + (1-x1)**2`
""")

if run:
    try:
        if not (0 < c1 < c2 < 1):
            st.error("Debe cumplirse 0 < c1 < c2 < 1 para las condiciones de Wolfe.")
            st.stop()

        expr, f, g, h = build_functions(function_text, int(n))
        x0 = parse_vector(x0_text, int(n))
        xmin, fmin, iters, history, stop_reason = optimize(method, f, g, h, x0, int(max_iter), tol, alpha0, c1, c2, rho)

        st.success("Optimización finalizada")
        col1, col2, col3 = st.columns(3)
        col1.metric("Punto mínimo encontrado", np.array2string(xmin, precision=6))
        col2.metric("Valor f(x*)", f"{fmin:.10g}")
        col3.metric("Iteraciones", iters)

        final_error = history[-1]["error ||grad||"]
        st.write(f"**Error final:** {final_error:.10g}")
        st.write(f"**Criterio de parada:** {stop_reason}")
        st.write(f"**Función interpretada:** `{sp.sstr(expr)}`")

        df = pd.DataFrame([
            {
                "iteración": item["iteración"],
                "f(x)": item["f(x)"],
                "error ||grad||": item["error ||grad||"],
                "x": np.array2string(item["x"], precision=6)
            }
            for item in history
        ])

        st.subheader("Tabla de iteraciones")
        st.dataframe(df, use_container_width=True)

        st.subheader("Gráfico de convergencia")
        fig, ax = plt.subplots()
        errores = df["error ||grad||"].values
        positivos = errores[errores > 0]
        usar_log = len(positivos) >= 2 and (errores.max() / positivos.min() > 100)
        if usar_log:
            ax.semilogy(df["iteración"], np.maximum(errores, 1e-16), marker="o")
            ax.set_ylabel("Error ||gradiente|| (escala log)")
        else:
            ax.plot(df["iteración"], errores, marker="o")
            ax.set_ylabel("Error ||gradiente||")
        ax.set_xlabel("Número de iteración")
        ax.set_title("Error versus número de iteraciones")
        ax.grid(True)
        st.pyplot(fig)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Descargar historial CSV", data=csv, file_name="historial_optimizacion.csv", mime="text/csv")

        st.info("Valor agregado: la aplicación permite descargar el historial de iteraciones en CSV para revisar el proceso paso a paso.")

    except Exception as e:
        st.error(str(e))
else:
    st.info("Completa los parámetros de entrada en la barra lateral y presiona Ejecutar optimización.")
