"""
AI-Assisted Momentum Management System — Orbital Skyhook Simulator
Streamlit UI

Mode 1: Physics Demonstration
    Single capture-release event. Shows angular velocity, energy, and
    tip-velocity degradation using a rigid-rod moment-of-inertia model.

Mode 2: Momentum Management & Operations Dashboard
    Long-term (days-to-years) simulation of launch/return traffic,
    momentum battery state, orbit evolution, and thruster/fuel costs.

Both modes share the same dark-console design system (cards, section
headers, Plotly layout) defined below.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from physics import SkyhookParams, run_simulation, sweep_payload_mass
from simulation_engine import run_momentum_management_sim, compute_kpis

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Skyhook Momentum Simulator",
    page_icon="🛰️",
    layout="wide",
)

# ── Custom CSS (shared design system for both modes) ─────────────────────────
st.markdown("""
<style>
  /* ---- global ---- */
  [data-testid="stAppViewContainer"] {
    background: #0b0f1a;
    color: #e8eaf0;
  }
  [data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid #1e2d45;
  }
  h1, h2, h3 { color: #e8eaf0; }

  /* ---- metric cards ---- */
  .card {
    background: #111827;
    border: 1px solid #1e2d45;
    border-radius: 10px;
    padding: 18px 20px 14px;
    margin-bottom: 12px;
  }
  .card-label {
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #5b7fa6;
    margin-bottom: 4px;
  }
  .card-value {
    font-size: 1.55rem;
    font-weight: 700;
    color: #57c7f5;
    font-family: 'Courier New', monospace;
  }
  .card-unit {
    font-size: 0.78rem;
    color: #5b7fa6;
    margin-left: 4px;
  }
  .card-delta {
    font-size: 0.78rem;
    margin-top: 4px;
  }
  .delta-warn  { color: #f0a347; }
  .delta-ok    { color: #4ecb85; }
  .delta-bad   { color: #e05c5c; }

  /* ---- section headers ---- */
  .section-title {
    font-size: 0.7rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #3a5a80;
    border-bottom: 1px solid #1e2d45;
    padding-bottom: 6px;
    margin: 24px 0 16px;
  }

  /* ---- alert banners ---- */
  .alert-box {
    background: #1a1008;
    border-left: 3px solid #f0a347;
    border-radius: 6px;
    padding: 10px 16px;
    font-size: 0.83rem;
    color: #c89050;
    margin-bottom: 16px;
  }
  .alert-box-ok {
    background: #0c1a10;
    border-left: 3px solid #4ecb85;
    border-radius: 6px;
    padding: 10px 16px;
    font-size: 0.83rem;
    color: #6fd99a;
    margin-bottom: 16px;
  }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def metric_card(label: str, value: str, unit: str = "", delta: str = "", delta_class: str = "delta-ok"):
    delta_html = f'<div class="card-delta {delta_class}">{delta}</div>' if delta else ""
    st.markdown(f"""
    <div class="card">
      <div class="card-label">{label}</div>
      <div class="card-value">{value}<span class="card-unit">{unit}</span></div>
      {delta_html}
    </div>
    """, unsafe_allow_html=True)


def fmt(val, decimals=4):
    return f"{val:,.{decimals}f}"


PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0b0f1a",
    plot_bgcolor="#0d1320",
    font=dict(color="#8aa3c0", size=11),
    margin=dict(l=50, r=20, t=40, b=50),
    xaxis=dict(gridcolor="#1a2540", zeroline=False),
    yaxis=dict(gridcolor="#1a2540", zeroline=False),
)

LINE_COLOR_1 = "#57c7f5"  # cyan
LINE_COLOR_2 = "#f0a347"  # amber
LINE_COLOR_3 = "#4ecb85"  # green
LINE_COLOR_4 = "#c07ef5"  # purple
LINE_COLOR_5 = "#e05c5c"  # red
FILL_ALPHA   = "rgba(87,199,245,0.08)"


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='margin-bottom:2px'>🛰️ Skyhook Momentum Management</h1>
<p style='color:#3a5a80;font-size:0.85rem;margin-top:0'>
AI-Assisted Orbital Tether Energy &amp; Momentum Simulator
</p>
""", unsafe_allow_html=True)


# ── Mode selector ─────────────────────────────────────────────────────────────
mode = st.radio(
    "Mode",
    ["Mode 1 — Physics Demonstration", "Mode 2 — Momentum Management & Operations"],
    horizontal=True,
    label_visibility="collapsed",
)


# ════════════════════════════════════════════════════════════════════════════
# MODE 1 — PHYSICS DEMONSTRATION
# ════════════════════════════════════════════════════════════════════════════
if mode.startswith("Mode 1"):

    # ── Sidebar inputs ───────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ System Parameters")
        st.markdown('<div class="section-title">Tether Assembly</div>', unsafe_allow_html=True)

        skyhook_mass = st.slider(
            "Skyhook Mass (kg)", min_value=500, max_value=50_000,
            value=10_000, step=500,
            help="Total mass of tether + hub structure"
        )
        tether_length = st.slider(
            "Tether Length (m)", min_value=500, max_value=20_000,
            value=5_000, step=250,
            help="Full tip-to-tip tether length"
        )

        st.markdown('<div class="section-title">Operational State</div>', unsafe_allow_html=True)

        initial_omega = st.slider(
            "Initial Angular Velocity ω₀ (rad/s)",
            min_value=0.001, max_value=0.05,
            value=0.012, step=0.001, format="%.3f",
            help="Rotational rate before payload capture"
        )
        payload_mass = st.slider(
            "Payload Mass (kg)", min_value=10, max_value=5_000,
            value=500, step=10,
            help="Mass of the cargo being transferred"
        )

        st.markdown('<div class="section-title">Graph Range</div>', unsafe_allow_html=True)
        mass_max = st.slider(
            "Payload mass sweep max (kg)", min_value=200, max_value=10_000,
            value=2_000, step=100
        )
        st.markdown("---")
        st.caption("Physics are simplified for educational / hackathon use.")

    # ── Run simulation ───────────────────────────────────────────────────────
    params = SkyhookParams(
        skyhook_mass=skyhook_mass,
        tether_length=tether_length,
        payload_mass=payload_mass,
        initial_omega=initial_omega,
    )
    res = run_simulation(params)

    mass_range = np.linspace(1, mass_max, 300)
    sweep = sweep_payload_mass(params, mass_range)

    # Alert if degradation is high
    if res.omega_degradation_pct > 15:
        st.markdown(f"""
        <div class="alert-box">
          ⚠️  High degradation detected — angular velocity drops by
          <strong>{res.omega_degradation_pct:.1f}%</strong>.
          Consider reducing payload mass or increasing skyhook mass ratio.
        </div>
        """, unsafe_allow_html=True)

    # ── Row 1: Angular velocity & energy ────────────────────────────────────
    st.markdown('<div class="section-title">Angular Velocity</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("Initial ω₀", fmt(res.initial_omega, 4), "rad/s")
    with c2:
        metric_card(
            "Final ω (after release)", fmt(res.omega_final, 4), "rad/s",
            delta=f"▼ {res.omega_degradation_pct:.2f}% drop",
            delta_class="delta-warn" if res.omega_degradation_pct > 10 else "delta-ok",
        )
    with c3:
        metric_card("Initial Tip Velocity", fmt(res.tip_velocity_initial, 2), "m/s")
    with c4:
        metric_card(
            "Final Tip Velocity", fmt(res.tip_velocity_final, 2), "m/s",
            delta=f"Payload Δv ≈ {res.delta_v_payload:.2f} m/s",
            delta_class="delta-ok",
        )

    st.markdown('<div class="section-title">Energy & Momentum</div>', unsafe_allow_html=True)
    c5, c6, c7, c8 = st.columns(4)

    with c5:
        metric_card("Initial Rotational KE", fmt(res.KE_initial / 1e6, 3), "MJ")
    with c6:
        metric_card(
            "Energy Lost", fmt(res.KE_lost / 1e6, 3), "MJ",
            delta=f"{res.KE_lost/res.KE_initial*100:.2f}% of initial KE",
            delta_class="delta-bad" if res.KE_lost/res.KE_initial > 0.2 else "delta-warn",
        )
    with c7:
        metric_card("Angular Momentum Transferred", fmt(res.L_transferred, 1), "kg·m²/s")
    with c8:
        metric_card(
            "Correction Impulse Required", fmt(res.correction_impulse / 1e3, 3), "kN·s",
            delta=f"≈ {res.correction_dv_skyhook:.3f} m/s Δv on skyhook",
            delta_class="delta-warn",
        )

    st.markdown('<div class="section-title">System Performance</div>', unsafe_allow_html=True)
    cp1, cp2, cp3 = st.columns(3)
    with cp1:
        metric_card(
            "Performance Degradation", fmt(res.performance_degradation_pct, 2), "%",
            delta="(tip velocity drop — delivery capacity proxy)",
            delta_class="delta-bad" if res.performance_degradation_pct > 10 else "delta-warn",
        )
    with cp2:
        ratio = payload_mass / skyhook_mass
        metric_card(
            "Payload / Skyhook Mass Ratio", fmt(ratio, 4), "",
            delta="Lower is better for momentum retention",
            delta_class="delta-ok" if ratio < 0.05 else "delta-warn",
        )
    with cp3:
        metric_card(
            "Moment of Inertia (skyhook)", fmt(res.I_skyhook / 1e6, 3), "× 10⁶ kg·m²"
        )

    # ── Graphs ───────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Sensitivity Analysis — Payload Mass Sweep</div>',
                unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Angular Velocity", "Energy Loss", "System Dashboard"])

    # Tab 1: ω vs payload mass
    with tab1:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=sweep["payload_mass"], y=sweep["omega_final"],
            mode="lines", name="Final ω",
            line=dict(color=LINE_COLOR_1, width=2.5),
            fill="tozeroy", fillcolor=FILL_ALPHA,
        ))
        fig1.add_hline(
            y=initial_omega, line_dash="dot", line_color="#4ecb85",
            annotation_text=f"ω₀ = {initial_omega:.3f} rad/s",
            annotation_font_color="#4ecb85",
        )
        fig1.add_vline(
            x=payload_mass, line_dash="dash", line_color="#f0a347",
            annotation_text=f"Current: {payload_mass} kg",
            annotation_font_color="#f0a347",
        )
        fig1.update_layout(
            **PLOTLY_LAYOUT,
            title="Final Angular Velocity vs Payload Mass",
            xaxis_title="Payload Mass (kg)",
            yaxis_title="ω final (rad/s)",
        )
        st.plotly_chart(fig1, use_container_width=True)

    # Tab 2: Energy loss
    with tab2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=sweep["payload_mass"], y=sweep["ke_lost_MJ"],
            mode="lines", name="KE Lost (MJ)",
            line=dict(color=LINE_COLOR_2, width=2.5),
            fill="tozeroy", fillcolor="rgba(240,163,71,0.07)",
        ))
        fig2.add_vline(
            x=payload_mass, line_dash="dash", line_color="#57c7f5",
            annotation_text=f"Current: {payload_mass} kg",
            annotation_font_color="#57c7f5",
        )
        fig2.update_layout(
            **PLOTLY_LAYOUT,
            title="Rotational Energy Lost vs Payload Mass",
            xaxis_title="Payload Mass (kg)",
            yaxis_title="Energy Lost (MJ)",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Tab 3: Combined dashboard
    with tab3:
        fig3 = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "ω Final vs Payload Mass",
                "Energy Lost (MJ)",
                "Performance Degradation (%)",
                "Correction Impulse (kN·s)",
            ],
        )

        traces = [
            (sweep["omega_final"],            LINE_COLOR_1, "ω Final (rad/s)",      1, 1),
            (sweep["ke_lost_MJ"],             LINE_COLOR_2, "KE Lost (MJ)",         1, 2),
            (sweep["perf_degradation_pct"],   LINE_COLOR_3, "Perf Degradation (%)", 2, 1),
            (sweep["correction_impulse_kNs"], LINE_COLOR_4, "Correction (kN·s)",    2, 2),
        ]
        for y_data, color, name, row, col in traces:
            fig3.add_trace(
                go.Scatter(
                    x=sweep["payload_mass"], y=y_data,
                    mode="lines", name=name,
                    line=dict(color=color, width=2),
                ),
                row=row, col=col,
            )
            fig3.add_vline(
                x=payload_mass, line_dash="dot", line_color="#ffffff",
                line_width=0.8, row=row, col=col,
            )

        fig3.update_layout(
            **PLOTLY_LAYOUT,
            title="Full System Dashboard",
            showlegend=True,
            height=560,
        )
        fig3.update_annotations(font_color="#8aa3c0", font_size=11)
        for ax in fig3.layout:
            if ax.startswith("xaxis") or ax.startswith("yaxis"):
                fig3.layout[ax].update(gridcolor="#1a2540", zeroline=False)

        st.plotly_chart(fig3, use_container_width=True)

    # ── ML Hook section ──────────────────────────────────────────────────────
    st.markdown('<div class="section-title">ML Integration Hook (raw output)</div>',
                unsafe_allow_html=True)

    ml_payload = {
        "inputs": {
            "skyhook_mass_kg":    res.skyhook_mass,
            "tether_length_m":    res.tether_length,
            "payload_mass_kg":    res.payload_mass,
            "initial_omega_rads": res.initial_omega,
        },
        "outputs": {
            "omega_final":                 res.omega_final,
            "omega_degradation_pct":       res.omega_degradation_pct,
            "ke_lost_J":                   res.KE_lost,
            "angular_momentum_transferred": res.L_transferred,
            "correction_impulse_Ns":       res.correction_impulse,
            "performance_degradation_pct": res.performance_degradation_pct,
            "tip_velocity_initial_ms":     res.tip_velocity_initial,
            "tip_velocity_final_ms":       res.tip_velocity_final,
            "delta_v_payload_ms":          res.delta_v_payload,
        },
    }

    with st.expander("View JSON — pipe this to your ML model"):
        st.json(ml_payload)

    st.caption("Physics model: simplified angular momentum conservation. Not for flight use.")


# ════════════════════════════════════════════════════════════════════════════
# MODE 2 — MOMENTUM MANAGEMENT & OPERATIONS DASHBOARD
# ════════════════════════════════════════════════════════════════════════════
else:

    # ── Sidebar inputs ───────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Mission Parameters")
        st.markdown('<div class="section-title">Tether Assembly</div>', unsafe_allow_html=True)

        tether_mass_2 = st.slider(
            "Tether Mass (kg)", min_value=1_000, max_value=200_000,
            value=50_000, step=1_000,
            help="Effective inertial mass used for momentum bookkeeping",
        )
        tether_length_2 = st.slider(
            "Tether Length (m)", min_value=500, max_value=30_000,
            value=10_000, step=250,
        )
        initial_altitude_km = st.slider(
            "Initial Orbital Altitude (km)", min_value=200, max_value=2_000,
            value=500, step=10,
        )

        st.markdown('<div class="section-title">Traffic Profile</div>', unsafe_allow_html=True)

        outgoing_mass = st.slider(
            "Outgoing Payload Mass (kg)", min_value=0, max_value=10_000,
            value=1_000, step=100,
            help="Mass per outgoing (launch) payload",
        )
        incoming_mass = st.slider(
            "Incoming Payload Mass (kg)", min_value=0, max_value=10_000,
            value=1_000, step=100,
            help="Mass per incoming (return) payload",
        )
        launches_per_day = st.slider(
            "Launches per Day", min_value=0.0, max_value=50.0,
            value=10.0, step=1.0,
        )
        returns_per_day = st.slider(
            "Returns per Day", min_value=0.0, max_value=50.0,
            value=10.0, step=1.0,
        )

        st.markdown('<div class="section-title">Mission Profile</div>', unsafe_allow_html=True)

        sim_days = st.select_slider(
            "Mission Duration (days)",
            options=[30, 90, 180, 365, 730, 1000],
            value=365,
        )
        thruster_efficiency = st.slider(
            "Thruster Correction Efficiency", min_value=0.0, max_value=1.0,
            value=0.5, step=0.05,
            help="Fraction of the daily momentum deficit restored by thrusters",
        )

        st.markdown("---")
        st.caption("Physics are simplified for educational / hackathon use.")

    # ── Run simulation ───────────────────────────────────────────────────────
    df2 = run_momentum_management_sim(
        tether_mass=tether_mass_2,
        tether_length=tether_length_2,
        initial_altitude=initial_altitude_km * 1000.0,
        outgoing_payload_mass=outgoing_mass,
        incoming_payload_mass=incoming_mass,
        launches_per_day=launches_per_day,
        returns_per_day=returns_per_day,
        simulation_days=sim_days,
        thruster_efficiency=thruster_efficiency,
    )
    k = compute_kpis(
        df2, launches_per_day, returns_per_day,
        outgoing_mass, incoming_mass, sim_days,
    )

    net_traffic = launches_per_day - returns_per_day

    # ── Status alert ─────────────────────────────────────────────────────────
    if k["battery_percent_final"] < 80:
        st.markdown(f"""
        <div class="alert-box">
          ⚠️  Momentum battery trending low — finishing at
          <strong>{k['battery_percent_final']:.1f}%</strong> after {sim_days} days.
          Increase returns/day or thruster correction efficiency.
        </div>
        """, unsafe_allow_html=True)
    elif net_traffic > 0 and k["battery_percent_final"] >= 80:
        st.markdown(f"""
        <div class="alert-box-ok">
          ✅  Net outgoing traffic ({net_traffic:+.1f}/day) is being offset by
          thruster correction — battery holding at
          <strong>{k['battery_percent_final']:.1f}%</strong>.
        </div>
        """, unsafe_allow_html=True)
    elif net_traffic < 0:
        st.markdown(f"""
        <div class="alert-box-ok">
          🔋  Net incoming traffic ({net_traffic:+.1f}/day) — momentum battery is
          charging, ending at <strong>{k['battery_percent_final']:.1f}%</strong>.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="alert-box-ok">
          ✅  Traffic balanced — momentum battery stable at
          <strong>{k['battery_percent_final']:.1f}%</strong>.
        </div>
        """, unsafe_allow_html=True)

    # ── Row 1: System status ─────────────────────────────────────────────────
    st.markdown('<div class="section-title">System Status</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        batt_delta_class = (
            "delta-ok" if k["battery_percent_change"] >= 0 else
            ("delta-bad" if k["battery_percent_final"] < 80 else "delta-warn")
        )
        metric_card(
            "Momentum Battery", fmt(k["battery_percent_final"], 1), "%",
            delta=f"{k['battery_percent_change']:+.2f}% vs. start",
            delta_class=batt_delta_class,
        )
    with c2:
        metric_card(
            "Net Momentum Balance", f"{k['net_momentum_balance']:.2e}", "m²/s",
            delta="recovered − lost",
            delta_class="delta-ok" if k["net_momentum_balance"] >= 0 else "delta-bad",
        )
    with c3:
        metric_card(
            "Fuel Consumed", fmt(k["fuel_consumed"], 1), "kg",
            delta=f"over {k['simulation_days']} days",
            delta_class="delta-warn",
        )
    with c4:
        eff_class = (
            "delta-ok" if k["system_efficiency"] >= 95 else
            ("delta-bad" if k["system_efficiency"] < 70 else "delta-warn")
        )
        metric_card(
            "System Efficiency", fmt(k["system_efficiency"], 1), "%",
            delta="recovery vs. loss",
            delta_class=eff_class,
        )

    # ── Row 2: Traffic summary ───────────────────────────────────────────────
    st.markdown('<div class="section-title">Traffic Summary</div>', unsafe_allow_html=True)
    c5, c6, c7, c8 = st.columns(4)

    with c5:
        metric_card(
            "Launches Completed", fmt(k["total_launches"], 0), "",
            delta=f"{launches_per_day:.1f} / day",
            delta_class="delta-warn",
        )
    with c6:
        metric_card(
            "Returns Completed", fmt(k["total_returns"], 0), "",
            delta=f"{returns_per_day:.1f} / day",
            delta_class="delta-ok",
        )
    with c7:
        metric_card(
            "Net Traffic", f"{k['net_traffic']:+,.0f}", "",
            delta="launches − returns",
            delta_class="delta-bad" if k["net_traffic"] > 0 else "delta-ok",
        )
    with c8:
        metric_card(
            "Mission Duration", fmt(k["simulation_days"], 0), "days",
            delta=f"{k['simulation_days']/365:.2f} yr",
            delta_class="delta-ok",
        )

    # ── Row 3: Orbit & propulsion ────────────────────────────────────────────
    st.markdown('<div class="section-title">Orbit &amp; Propulsion</div>', unsafe_allow_html=True)
    c9, c10, c11, c12 = st.columns(4)

    with c9:
        metric_card("Initial Altitude", fmt(k["initial_altitude_km"], 2), "km")
    with c10:
        metric_card("Final Altitude", fmt(k["final_altitude_km"], 2), "km")
    with c11:
        metric_card(
            "Altitude Change", f"{k['altitude_change_km']:+.3f}", "km",
            delta="net over mission",
            delta_class="delta-ok" if k["altitude_change_km"] >= 0 else "delta-bad",
        )
    with c12:
        metric_card(
            "Thruster Corrections Applied", f"{k['total_thruster_corrections']:.2e}", "m²/s",
            delta="cumulative",
            delta_class="delta-warn",
        )

    # ── Row 4: Momentum & fuel engineering ───────────────────────────────────
    st.markdown('<div class="section-title">Momentum &amp; Fuel Engineering</div>', unsafe_allow_html=True)
    c13, c14, c15, c16 = st.columns(4)

    with c13:
        metric_card(
            "Momentum Lost (Launches)", f"{k['momentum_lost_total']:.2e}", "m²/s",
            delta="cumulative",
            delta_class="delta-bad",
        )
    with c14:
        metric_card(
            "Momentum Recovered (Returns)", f"{k['momentum_recovered_total']:.2e}", "m²/s",
            delta="cumulative",
            delta_class="delta-ok",
        )
    with c15:
        metric_card(
            "Avg Daily Momentum Deficit", f"{k['avg_daily_deficit']:.2e}", "m²/s/day",
            delta="pre-correction",
            delta_class="delta-warn",
        )
    with c16:
        metric_card(
            "Total Momentum Deficit", f"{k['total_momentum_deficit']:.2e}", "m²/s",
            delta="pre-correction sum",
            delta_class="delta-warn",
        )

    c17, c18, c19 = st.columns(3)
    with c17:
        metric_card(
            "Fuel per Launch", fmt(k["fuel_per_launch"], 2), "kg",
            delta="avg. across mission",
            delta_class="delta-warn",
        )
    with c18:
        metric_card(
            "Fuel per kg Payload", fmt(k["fuel_per_kg_payload"], 4), "kg/kg",
            delta="outgoing payload basis",
            delta_class="delta-warn",
        )
    with c19:
        metric_card(
            "Avg Momentum Recovery Rate", f"{k['avg_recovery_rate']:.2e}", "m²/s/day",
            delta="returns + thrusters",
            delta_class="delta-ok",
        )

    # ── Graphs ───────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Telemetry — Mission Timeline</div>',
                unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Momentum Battery", "Orbital Altitude", "Fuel Consumption", "Daily Momentum Balance"]
    )

    # Tab 1: Momentum Battery vs Time
    with tab1:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=df2["day"], y=df2["battery_percent"],
            mode="lines", name="Battery (%)",
            line=dict(color=LINE_COLOR_1, width=2.5),
            fill="tozeroy", fillcolor=FILL_ALPHA,
        ))
        fig1.add_hline(
            y=100, line_dash="dot", line_color="#4ecb85",
            annotation_text="100% nominal",
            annotation_font_color="#4ecb85",
        )
        fig1.add_hline(
            y=0, line_dash="dot", line_color="#e05c5c",
            annotation_text="0% depleted",
            annotation_font_color="#e05c5c",
        )
        fig1.update_layout(
            **PLOTLY_LAYOUT,
            title="Momentum Battery vs Time",
            xaxis_title="Day",
            yaxis_title="Battery (%)",
        )
        st.plotly_chart(fig1, use_container_width=True)

    # Tab 2: Orbital Altitude vs Time
    with tab2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df2["day"], y=df2["altitude_km"],
            mode="lines", name="Altitude (km)",
            line=dict(color=LINE_COLOR_2, width=2.5),
            fill="tozeroy", fillcolor="rgba(240,163,71,0.07)",
        ))
        fig2.update_layout(
            **PLOTLY_LAYOUT,
            title="Orbital Altitude vs Time",
            xaxis_title="Day",
            yaxis_title="Altitude (km)",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Tab 3: Fuel Consumption vs Time
    with tab3:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=df2["day"], y=df2["cumulative_fuel_use"],
            mode="lines", name="Cumulative Fuel (kg)",
            line=dict(color=LINE_COLOR_4, width=2.5),
            fill="tozeroy", fillcolor="rgba(192,126,245,0.08)",
        ))
        fig3.update_layout(
            **PLOTLY_LAYOUT,
            title="Cumulative Fuel Consumption vs Time",
            xaxis_title="Day",
            yaxis_title="Fuel Used (kg)",
        )
        st.plotly_chart(fig3, use_container_width=True)

    # Tab 4: Daily Momentum Balance vs Time
    with tab4:
        colors = np.where(df2["daily_momentum_balance"] >= 0, LINE_COLOR_3, LINE_COLOR_5)
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(
            x=df2["day"], y=df2["daily_momentum_balance"],
            name="Daily Momentum Balance",
            marker_color=colors,
        ))
        fig4.add_hline(y=0, line_color="#8aa3c0", line_width=1)
        fig4.update_layout(
            **PLOTLY_LAYOUT,
            title="Daily Momentum Balance vs Time (Recovered − Lost)",
            xaxis_title="Day",
            yaxis_title="Δ Momentum (m²/s)",
        )
        st.plotly_chart(fig4, use_container_width=True)

    # ── ML Hook section ──────────────────────────────────────────────────────
    st.markdown('<div class="section-title">ML Integration Hook (raw output)</div>',
                unsafe_allow_html=True)

    ml_payload_2 = {
        "inputs": {
            "tether_mass_kg":            tether_mass_2,
            "tether_length_m":           tether_length_2,
            "initial_altitude_km":       initial_altitude_km,
            "outgoing_payload_mass_kg":  outgoing_mass,
            "incoming_payload_mass_kg":  incoming_mass,
            "launches_per_day":          launches_per_day,
            "returns_per_day":           returns_per_day,
            "simulation_days":           sim_days,
            "thruster_efficiency":       thruster_efficiency,
        },
        "outputs": k,
    }

    with st.expander("View JSON — pipe this to your ML model"):
        st.json(ml_payload_2)

    # ── Raw data ──────────────────────────────────────────────────────────────
    with st.expander("📄 Raw simulation data (daily log)"):
        st.dataframe(df2, use_container_width=True)

    st.caption("Physics model: simplified momentum-battery / angular-momentum conservation. Not for flight use.")
