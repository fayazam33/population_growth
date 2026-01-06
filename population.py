import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ----------------- Logistic ODE Functions -----------------
def logistic_ode(P, r, K):
    return r * P * (1 - P / K)

def euler_method(P0, r, K, h, steps):
    P = P0
    populations = [P]
    for _ in range(steps):
        P = P + h * logistic_ode(P, r, K)
        populations.append(P)
    return np.array(populations)

def rk4_method(P0, r, K, h, steps):
    P = P0
    populations = [P]
    for _ in range(steps):
        k1 = logistic_ode(P, r, K)
        k2 = logistic_ode(P + 0.5*h*k1, r, K)
        k3 = logistic_ode(P + 0.5*h*k2, r, K)
        k4 = logistic_ode(P + h*k3, r, K)
        P = P + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        populations.append(P)
    return np.array(populations)

# ----------------- Logistic Regression Fit -----------------
def logistic_function(t, r, K, A):
    return K / (1 + A * np.exp(-r*t))

def fit_logistic(t_data, p_data):
    initial_guess = [0.2, 400, 5]
    params, _ = curve_fit(logistic_function, t_data, p_data, p0=initial_guess, maxfev=5000)
    return params

# ----------------- Individual Resource Effects -----------------
def individual_resource_effect(food=0.0, water=0.0, medicine=0.0):
    r_multiplier = 1 + 0.5 * water
    K_multiplier = 1 + 0.7 * food
    shock_reduction = 1 - 0.8 * medicine
    return r_multiplier, K_multiplier, shock_reduction

# ----------------- Shock Loss Function -----------------
def shock_loss(P, K, shock_type, shock_reduction=1.0):
    if shock_type == "pandemic":
        alpha = 0.4
    elif shock_type == "war":
        alpha = 0.7
    elif shock_type == "environment":
        alpha = 0.25
    else:
        alpha = 0.3
    loss = alpha * (P / K) * P
    return loss * shock_reduction

# ----------------- Span Shock Simulation -----------------
def span_shock_simulation(
    P0, r, K, h, days,
    shock_start, shock_end,
    shock_type,
    food=0.0,
    water=0.0,
    medicine=0.0
):
    P = P0
    populations = []

    r_mul, K_mul, shock_red = individual_resource_effect(food, water, medicine)
    r_eff = r * r_mul
    K_eff = K * K_mul

    for day in range(days + 1):
        if shock_start <= day <= shock_end:
            loss = shock_loss(P, K_eff, shock_type, shock_red)
        else:
            loss = 0

        populations.append(P)
        P = P + h * logistic_ode(P, r_eff, K_eff) - loss

    return np.array(populations)

# ----------------- Streamlit Interface -----------------
st.title("Population Growth Simulation with Shocks & Resources")


st.sidebar.header("Simulation Parameters")
P0 = st.sidebar.number_input("Initial Population (P0)", value=50)
r = st.sidebar.number_input("Growth Rate (r)", value=0.3, step=0.01)
K = st.sidebar.number_input("Carrying Capacity (K)", value=500)
days = st.sidebar.slider("Simulation Days", 50, 1000, 500)

st.sidebar.header("Shock Parameters")
shock_type = st.sidebar.selectbox("Shock Type", ["pandemic", "war", "environment"])
shock_start = st.sidebar.slider("Shock Start Day", 0, days-1, 15)
shock_end = st.sidebar.slider("Shock End Day", shock_start+1, days, 20)

st.sidebar.header("Resource Levels (0 to 1)")
food = st.sidebar.slider("Food", 0.0, 1.0, 0.3)
water = st.sidebar.slider("Water", 0.0, 1.0, 0.3)
medicine = st.sidebar.slider("Medicine", 0.0, 1.0, 0.3)

# ----------------- Simulations -----------------
time = np.arange(days + 1)
rk4_pop = rk4_method(P0, r, K, 1, days)
euler_pop = euler_method(P0, r, K, 1, days)

# Logistic regression fit
r_est, K_est, A_est = fit_logistic(time, rk4_pop)
p_fit = logistic_function(time, r_est, K_est, A_est)

# Shock + resources
shock_pop = span_shock_simulation(
    P0, r_est, K_est, 1, days,
    shock_start, shock_end,
    shock_type, food, water, medicine
)

# ----------------- Plots -----------------
st.subheader("Euler vs RK4")
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(time, rk4_pop, label="RK4", color='blue')
ax.plot(time, euler_pop, label="Euler", color='red', linestyle='--')
ax.set_xlabel("Time")
ax.set_ylabel("Population")
ax.legend()
ax.grid(True)
st.pyplot(fig)

st.subheader("Absolute Error |RK4 − Euler|")
error = np.abs(rk4_pop - euler_pop)
fig_err, ax_err = plt.subplots(figsize=(8,4))
ax_err.plot(time, error, color='red', linewidth=2)
ax_err.set_xlabel("Time")
ax_err.set_ylabel("Absolute Error")
ax_err.grid(True)
st.pyplot(fig_err)

st.subheader("Logistic Regression Fit to RK4")
fig2, ax2 = plt.subplots(figsize=(8,5))
ax2.scatter(time, rk4_pop, color="cyan", label="RK4 Data")
ax2.plot(time, p_fit, color="red", label="Regression Fit")
ax2.set_xlabel("Time")
ax2.set_ylabel("Population")
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)

st.subheader("Population with Shock + Resources")
fig3, ax3 = plt.subplots(figsize=(9,5))
ax3.plot(time, rk4_pop, label="Normal Growth (RK4)", linewidth=2)
ax3.plot(time, shock_pop, '--', label=f"{shock_type.capitalize()} + Resources", linewidth=2)
ax3.axvspan(shock_start, shock_end, color='red', alpha=0.2, label="Shock Period")
ax3.set_xlabel("Time")
ax3.set_ylabel("Population")
ax3.legend()
ax3.grid(True)
st.pyplot(fig3)


st.markdown(
    """
    <style>
    [data-testid="stSidebar"] { background-color: #6AECE1; }
    </style>
    """,
    unsafe_allow_html=True
)



