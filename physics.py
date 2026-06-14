"""Short-timescale rotational physics for the Skyhook dashboard."""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SkyhookParams:
    """Inputs for a single payload capture and release event."""

    skyhook_mass: float
    tether_length: float
    payload_mass: float
    initial_omega: float

    def __post_init__(self):
        for name in ("skyhook_mass", "tether_length", "initial_omega"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if self.payload_mass < 0:
            raise ValueError("payload_mass cannot be negative")


@dataclass(frozen=True)
class SimulationResult:
    skyhook_mass: float
    tether_length: float
    payload_mass: float
    initial_omega: float
    omega_final: float
    omega_degradation_pct: float
    tip_velocity_initial: float
    tip_velocity_final: float
    delta_v_payload: float
    I_skyhook: float
    KE_initial: float
    KE_lost: float
    L_transferred: float
    correction_impulse: float
    correction_dv_skyhook: float
    performance_degradation_pct: float


def run_simulation(params: SkyhookParams) -> SimulationResult:
    """Simulate a payload coupled to the tip of a uniform rigid tether."""
    tip_radius = params.tether_length / 2.0
    inertia = params.skyhook_mass * params.tether_length**2 / 12.0
    payload_inertia = params.payload_mass * tip_radius**2

    # Angular momentum is conserved while the initially stationary payload
    # couples to the rotating tether.
    omega_final = (
        inertia * params.initial_omega / (inertia + payload_inertia)
    )
    tip_velocity_initial = params.initial_omega * tip_radius
    tip_velocity_final = omega_final * tip_radius
    degradation_pct = (
        (params.initial_omega - omega_final) / params.initial_omega * 100.0
    )

    ke_initial = 0.5 * inertia * params.initial_omega**2
    ke_final_system = 0.5 * (inertia + payload_inertia) * omega_final**2
    ke_lost = max(ke_initial - ke_final_system, 0.0)
    angular_momentum_transferred = payload_inertia * omega_final
    correction_impulse = (
        angular_momentum_transferred / tip_radius if tip_radius else 0.0
    )

    return SimulationResult(
        skyhook_mass=params.skyhook_mass,
        tether_length=params.tether_length,
        payload_mass=params.payload_mass,
        initial_omega=params.initial_omega,
        omega_final=omega_final,
        omega_degradation_pct=degradation_pct,
        tip_velocity_initial=tip_velocity_initial,
        tip_velocity_final=tip_velocity_final,
        delta_v_payload=tip_velocity_final,
        I_skyhook=inertia,
        KE_initial=ke_initial,
        KE_lost=ke_lost,
        L_transferred=angular_momentum_transferred,
        correction_impulse=correction_impulse,
        correction_dv_skyhook=correction_impulse / params.skyhook_mass,
        performance_degradation_pct=degradation_pct,
    )


def sweep_payload_mass(
    params: SkyhookParams, payload_masses
) -> pd.DataFrame:
    """Run the single-event model over a one-dimensional mass range."""
    masses = np.asarray(payload_masses, dtype=float)
    if masses.ndim != 1:
        raise ValueError("payload_masses must be one-dimensional")
    if np.any(masses < 0):
        raise ValueError("payload masses cannot be negative")

    rows = []
    for mass in masses:
        result = run_simulation(replace(params, payload_mass=float(mass)))
        rows.append(
            {
                "payload_mass": mass,
                "omega_final": result.omega_final,
                "ke_lost_MJ": result.KE_lost / 1.0e6,
                "perf_degradation_pct": result.performance_degradation_pct,
                "correction_impulse_kNs": result.correction_impulse / 1.0e3,
            }
        )
    return pd.DataFrame(rows)
