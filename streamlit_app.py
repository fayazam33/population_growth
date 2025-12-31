import streamlit as st
import numpy as np
import pandas as pd
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
    return populations

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
    return populations

# ----------------- Logistic Regression Fit -----------------
def logistic_function(t, r, K, A):
    return K / (1 + A * np.exp(-r*t))

def fit_logistic(t_data, p_data):
    initial_guess = [0.2, 400, 5]
    params, _ = curve_fit(logistic_function, t_data, p_data, p0=initial_guess, maxfev=5000)
    return params

# ----------------- Pandemic/War Simulation -----------------
def shock_simulation(P0, r, K, h, days, shock_day, drop_fraction):
    P = P0
    populations = []
    for day in range(days + 1):
        if day == shock_day:
            P = P * (1 - drop_fraction)
        populations.append(P)
        P = P + h * logistic_ode(P, r, K)
    return populations

# ----------------- Streamlit Interface -----------------
st.title("Population Growth Simulation")

# Parameters
P0 = st.sidebar.number_input("Initial Population (P0)", value=50)
r = st.sidebar.number_input("Growth Rate (r)", value=0.3, step=0.01)
K = st.sidebar.number_input("Carrying Capacity (K)", value=500)
days = st.sidebar.slider("Simulation Days", min_value=50, max_value=1000, value=500)

# RK4 and Euler
time = np.arange(0, days+1)
rk4_pop = rk4_method(P0, r, K, 1, days)
euler_pop = euler_method(P0, r, K, 1, days)

st.subheader("Euler vs RK4")
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(time, rk4_pop, label="RK4", color='blue')
ax.plot(time, euler_pop, label="Euler", color='red', linestyle='--')
ax.set_xlabel("Time")
ax.set_ylabel("Population")
ax.legend()
st.pyplot(fig)

# Logistic Regression Fit
r_est, K_est, A_est = fit_logistic(time, rk4_pop)
t_fine = np.linspace(0, days, 200)
p_fit = logistic_function(t_fine, r_est, K_est, A_est)

st.subheader("Logistic Regression Fit to RK4 Data")
fig2, ax2 = plt.subplots(figsize=(8,5))
ax2.scatter(time, rk4_pop, color="cyan", label="RK4 Data")
ax2.plot(t_fine, p_fit, color="red", label="Regression Fit")
ax2.set_xlabel("Time")
ax2.set_ylabel("Population")
ax2.legend()
st.pyplot(fig2)

# Interactive Shock Simulation
st.subheader("Shock Simulation: Pandemic / War")
shock_day = st.slider("Shock Day", 0, days, 15)
drop_fraction = st.slider("Population Drop Fraction", 0.0, 1.0, 0.3, step=0.05)
shock_type = st.selectbox("Type of Shock", ["Pandemic", "War"])

shock_pop = shock_simulation(P0, r_est, K_est, 1, days, shock_day, drop_fraction)

fig3, ax3 = plt.subplots(figsize=(8,5))
ax3.plot(time, rk4_pop, label="Normal Growth (RK4)")
ax3.plot(time, shock_pop, '--', label=f"{shock_type}: {int(drop_fraction*100)}% drop on day {shock_day}")
ax3.axvline(shock_day, color="green", linestyle=":", label="Shock Day")
ax3.set_xlabel("Time")
ax3.set_ylabel("Population")
ax3.legend()
st.pyplot(fig3)

st.markdown(
    """
    <style>
    /* Sidebar background color */
    [data-testid="stSidebar"] {
        background-color: #6AECE1;
    }
    </style>
    """,
    unsafe_allow_html=True
)



