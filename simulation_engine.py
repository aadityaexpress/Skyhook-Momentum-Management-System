"""
simulation_engine.py

Long-term momentum management simulation engine for the Skyhook
Momentum Management System (Mode 2 — Operations Dashboard).

Contains:
    - run_momentum_management_sim: Long-term (days to years) simulation
        of a Skyhook operating as a "momentum battery", balancing
        outgoing launches against incoming returns, with thruster
        compensation and fuel-usage estimates.
    - compute_kpis: Derives summary KPIs from a simulation run for
        dashboard display.

Mode 1 (short-timescale physics demonstration) is now handled by
utils/physics.py, which uses an independent angular-momentum /
moment-of-inertia model centered on a single capture-release event.

Both engines in this file build on the shared physics primitives in
momentum_model.py.
"""

import numpy as np
import pandas as pd

from momentum_model import (
    SkyhookState,
    apply_payload_event,
    EARTH_MU,
)


# ----------------------------------------------------------------------
# Simulator V2 -- Momentum Management Simulator (Mode 2)
# ----------------------------------------------------------------------
def run_momentum_management_sim(tether_mass, tether_length, initial_altitude,
                                  outgoing_payload_mass, incoming_payload_mass,
                                  launches_per_day, returns_per_day,
                                  simulation_days, thruster_efficiency,
                                  delta_v_tether=2.0):
    """
    Run the long-term momentum management simulation.

    Each simulated day:
        1. Apply `launches_per_day` outgoing events (discharge battery).
        2. Apply `returns_per_day` incoming events (recharge battery).
        3. Compute the resulting "momentum deficit" relative to the
           nominal (100%) level.
        4. Estimate thruster correction effort needed to compensate for
           that deficit, scaled by `thruster_efficiency`.
        5. Accumulate estimated fuel usage.
        6. Recompute altitude from the updated angular momentum.

    Args:
        tether_mass (float): Tether system mass (kg).
        tether_length (float): Tether length (m), passed through for
            display only.
        initial_altitude (float): Starting altitude above Earth's
            surface (m).
        outgoing_payload_mass (float): Mass per outgoing payload (kg).
        incoming_payload_mass (float): Mass per incoming payload (kg).
        launches_per_day (float): Number of outgoing payload events per
            day (can be fractional to represent averages).
        returns_per_day (float): Number of incoming payload events per
            day.
        simulation_days (int): Total number of days to simulate.
        thruster_efficiency (float): Fraction in [0, 1] representing how
            efficiently thruster corrections restore the momentum
            battery. 1.0 = perfect correction (fully restores deficit
            each day, at proportional fuel cost). 0.0 = no correction
            (deficit accumulates unchecked, no fuel used).
        delta_v_tether (float): Per-event tether-side velocity change
            (m/s) used for both outgoing and incoming events. Default is
            a representative small value for a hackathon-scale model.

    Returns:
        pandas.DataFrame: One row per simulated day, with columns:
            - day (int)
            - angular_momentum (float, m^2/s)
            - battery_percent (float, %)
            - altitude_m (float, m)
            - altitude_km (float, km)
            - net_launches (float) -- launches minus returns that day
            - momentum_deficit (float, m^2/s) -- shortfall vs nominal
              BEFORE thruster correction
            - thruster_correction (float, m^2/s) -- angular momentum
              restored by thrusters this day
            - daily_fuel_use (float, kg) -- estimated fuel consumed this
              day
            - cumulative_fuel_use (float, kg) -- running total fuel used
    """
    state = SkyhookState(tether_mass, tether_length, initial_altitude)
    thruster_efficiency = float(np.clip(thruster_efficiency, 0.0, 1.0))

    # Fuel-use scaling constant: converts angular-momentum correction
    # (m^2/s) into an equivalent propellant mass (kg). This is a
    # simplified proportionality constant chosen so that fuel usage
    # scales sensibly with tether mass and correction magnitude --
    # it is NOT derived from a rocket-equation solve, consistent with
    # the project's "physically reasonable approximation" requirement.
    FUEL_SCALING = state.tether_mass / 1.0e6  # kg fuel per (m^2/s) of correction

    records = []
    cumulative_fuel = 0.0

    # Day 0 baseline
    records.append({
        "day": 0,
        "angular_momentum": state.angular_momentum,
        "battery_percent": state.battery_percent(),
        "altitude_m": state.altitude,
        "altitude_km": state.altitude / 1000.0,
        "net_launches": 0.0,
        "momentum_lost": 0.0,
        "momentum_recovered": 0.0,
        "daily_momentum_balance": 0.0,
        "momentum_deficit": 0.0,
        "thruster_correction": 0.0,
        "daily_fuel_use": 0.0,
        "cumulative_fuel_use": 0.0,
    })

    for day in range(1, int(simulation_days) + 1):
        # --- 1. Outgoing launches discharge the battery ---
        momentum_lost = 0.0
        if launches_per_day > 0:
            delta_L_out = apply_payload_event(
                state,
                outgoing_payload_mass * launches_per_day,
                delta_v_tether,
                direction="outgoing",
            )
            momentum_lost = abs(delta_L_out)

        # --- 2. Incoming returns recharge the battery ---
        momentum_recovered = 0.0
        if returns_per_day > 0:
            delta_L_in = apply_payload_event(
                state,
                incoming_payload_mass * returns_per_day,
                delta_v_tether,
                direction="incoming",
            )
            momentum_recovered = abs(delta_L_in)

        # --- 3. Momentum deficit relative to nominal (100%) ---
        deficit = state.angular_momentum_nominal - state.angular_momentum
        # Only a positive deficit (below nominal) triggers correction;
        # an "overcharged" battery (deficit < 0) is left as-is.
        deficit_to_correct = max(deficit, 0.0)

        # --- 4. Thruster correction restores part of the deficit ---
        thruster_correction = deficit_to_correct * thruster_efficiency
        state.angular_momentum += thruster_correction

        # --- 5. Fuel usage proportional to correction magnitude ---
        daily_fuel = thruster_correction * FUEL_SCALING
        cumulative_fuel += daily_fuel

        # --- 6. Recompute altitude after correction ---
        state.update_altitude_from_angular_momentum()

        records.append({
            "day": day,
            "angular_momentum": state.angular_momentum,
            "battery_percent": state.battery_percent(),
            "altitude_m": state.altitude,
            "altitude_km": state.altitude / 1000.0,
            "net_launches": launches_per_day - returns_per_day,
            "momentum_lost": momentum_lost,
            "momentum_recovered": momentum_recovered,
            "daily_momentum_balance": momentum_recovered - momentum_lost,
            "momentum_deficit": deficit,
            "thruster_correction": thruster_correction,
            "daily_fuel_use": daily_fuel,
            "cumulative_fuel_use": cumulative_fuel,
        })

    return pd.DataFrame(records)


def compute_kpis(df, launches_per_day, returns_per_day,
                  outgoing_payload_mass, incoming_payload_mass,
                  simulation_days):
    """
    Derive summary KPIs from a Mode 2 simulation DataFrame for display on
    an operations dashboard.

    Args:
        df (pandas.DataFrame): Output of run_momentum_management_sim.
        launches_per_day (float): Launches per day (as configured).
        returns_per_day (float): Returns per day (as configured).
        outgoing_payload_mass (float): Mass per outgoing payload (kg).
        incoming_payload_mass (float): Mass per incoming payload (kg).
        simulation_days (int): Total simulated days.

    Returns:
        dict: Summary KPIs, including totals, rates, and deltas suitable
        for direct display in KPI cards.
    """
    initial = df.iloc[0]
    final = df.iloc[-1]

    total_launches = launches_per_day * simulation_days
    total_returns = returns_per_day * simulation_days
    net_traffic = total_launches - total_returns

    total_momentum_lost = df["momentum_lost"].sum()
    total_momentum_recovered = df["momentum_recovered"].sum()
    net_momentum_balance = total_momentum_recovered - total_momentum_lost

    total_fuel = final["cumulative_fuel_use"]
    total_thruster_corrections = df["thruster_correction"].sum()
    total_momentum_deficit = df["momentum_deficit"].clip(lower=0).sum()
    avg_daily_deficit = (
        df["momentum_deficit"].clip(lower=0).iloc[1:].mean()
        if len(df) > 1 else 0.0
    )

    fuel_per_launch = (
        total_fuel / total_launches if total_launches > 0 else 0.0
    )
    total_outgoing_mass = outgoing_payload_mass * total_launches
    fuel_per_kg_payload = (
        total_fuel / total_outgoing_mass if total_outgoing_mass > 0 else 0.0
    )

    # System efficiency: how much of the momentum lost to launches was
    # recovered (via returns + thruster corrections), capped at 100%.
    total_recovery = total_momentum_recovered + total_thruster_corrections
    if total_momentum_lost > 0:
        system_efficiency = min(total_recovery / total_momentum_lost, 1.5) * 100.0
    else:
        system_efficiency = 100.0

    # Average momentum recovery rate (m^2/s per day) from returns + thrusters
    avg_recovery_rate = (
        total_recovery / simulation_days if simulation_days > 0 else 0.0
    )

    return {
        "battery_percent_final": final["battery_percent"],
        "battery_percent_change": final["battery_percent"] - initial["battery_percent"],
        "net_momentum_balance": net_momentum_balance,
        "fuel_consumed": total_fuel,
        "system_efficiency": system_efficiency,

        "total_launches": total_launches,
        "total_returns": total_returns,
        "net_traffic": net_traffic,
        "simulation_days": simulation_days,

        "initial_altitude_km": initial["altitude_km"],
        "final_altitude_km": final["altitude_km"],
        "altitude_change_km": final["altitude_km"] - initial["altitude_km"],
        "total_thruster_corrections": total_thruster_corrections,

        "momentum_lost_total": total_momentum_lost,
        "momentum_recovered_total": total_momentum_recovered,
        "avg_daily_deficit": avg_daily_deficit,
        "total_momentum_deficit": total_momentum_deficit,
        "fuel_per_launch": fuel_per_launch,
        "fuel_per_kg_payload": fuel_per_kg_payload,
        "avg_recovery_rate": avg_recovery_rate,
    }
