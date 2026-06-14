"""
momentum_model.py

Core physics primitives for the Skyhook Momentum Management System.

This module defines:
    - Constants
    - SkyhookState: the physical/operational state of the tether system
    - Momentum battery calculations
    - Single-event momentum transfer (used by both Mode 1 and Mode 2)
    - Simple orbital altitude response to angular momentum change

NOTE ON FIDELITY:
This is intentionally a *reduced-order* model suitable for a hackathon
demo. We treat the Skyhook as a point-mass / rigid-body proxy with an
"angular momentum reservoir" (L_tether). Payload interactions exchange
angular momentum with this reservoir via simple conservation:

    L_tether_after = L_tether_before - delta_L_payload

Orbital altitude is then nudged using a linearized relationship between
specific angular momentum and circular orbit radius:

    L_circular = sqrt(mu * r)  =>  r = L^2 / mu

This is "physically reasonable" (correct functional form for circular
orbits) without requiring full two-body / tether dynamics integration.
"""

import numpy as np

# ----------------------------------------------------------------------
# Physical constants
# ----------------------------------------------------------------------
EARTH_MU = 3.986004418e14      # Standard gravitational parameter of Earth, m^3/s^2
EARTH_RADIUS = 6.371e6          # Mean Earth radius, m


class SkyhookState:
    """
    Represents the current physical/operational state of the Skyhook
    tether system.

    Attributes:
        tether_mass (float): Mass of the tether system (kg). Used as the
            effective inertial mass for momentum bookkeeping.
        tether_length (float): Length of the tether (m). Currently used
            for context/display and future rotational-dynamics extensions.
        altitude (float): Current orbital altitude of the Skyhook center
            of mass above Earth's surface (m).
        angular_momentum (float): Specific angular momentum reservoir of
            the Skyhook (m^2/s). This is the "momentum battery" quantity.
        angular_momentum_nominal (float): The reference (100% charge)
            specific angular momentum, corresponding to the initial
            orbital altitude. Used to normalize the battery percentage.
        angular_momentum_min (float): The reservoir level corresponding
            to 0% charge (defined as a fraction of nominal, representing
            the "depleted" threshold before the orbit becomes unusable).
    """

    def __init__(self, tether_mass, tether_length, initial_altitude,
                 depletion_fraction=0.85):
        """
        Args:
            tether_mass (float): Tether system mass in kg.
            tether_length (float): Tether length in m.
            initial_altitude (float): Initial altitude above Earth's
                surface in meters.
            depletion_fraction (float): Fraction of nominal specific
                angular momentum that defines "0% battery" (e.g. 0.85
                means the battery is empty when L drops to 85% of its
                starting value). This keeps the orbit physically valid
                (r stays positive and reasonable) across the full 0-100%
                battery range.
        """
        self.tether_mass = tether_mass
        self.tether_length = tether_length
        self.altitude = initial_altitude

        r0 = EARTH_RADIUS + initial_altitude
        self.angular_momentum_nominal = circular_specific_angular_momentum(r0)
        self.angular_momentum = self.angular_momentum_nominal
        self.angular_momentum_min = self.angular_momentum_nominal * depletion_fraction

    def battery_percent(self):
        """
        Return the current momentum battery level as a percentage.

        100% corresponds to angular_momentum_nominal (initial state).
        0% corresponds to angular_momentum_min (depletion threshold).
        Values are clipped to [0, 100] for display purposes, though the
        underlying angular_momentum value is allowed to go slightly
        outside this range (battery can be "overcharged" above 100% if
        more momentum is returned than was originally present).
        """
        span = self.angular_momentum_nominal - self.angular_momentum_min
        if span <= 0:
            return 100.0
        pct = (self.angular_momentum - self.angular_momentum_min) / span * 100.0
        return float(np.clip(pct, 0.0, 150.0))  # allow modest overcharge display

    def update_altitude_from_angular_momentum(self):
        """
        Recompute the Skyhook's altitude based on its current specific
        angular momentum, assuming a circular-orbit relationship:

            L = sqrt(mu * r)  =>  r = L^2 / mu

        This is a simplification: a real tether system's center-of-mass
        orbit evolution depends on tether rotation, mass distribution,
        and attitude. For this educational simulator, we treat the whole
        system as if it were a point mass whose orbital radius responds
        directly to changes in specific angular momentum.
        """
        L = max(self.angular_momentum, 1.0)  # avoid non-physical negative/zero L
        r = (L ** 2) / EARTH_MU
        self.altitude = r - EARTH_RADIUS


def circular_specific_angular_momentum(r):
    """
    Specific angular momentum (per unit mass) for a circular orbit of
    radius r.

    L = sqrt(mu * r)

    Args:
        r (float): Orbital radius from Earth's center, in meters.

    Returns:
        float: Specific angular momentum, m^2/s.
    """
    return np.sqrt(EARTH_MU * r)


def payload_specific_angular_momentum_delta(payload_mass, tether_mass,
                                              delta_v_tether):
    """
    Estimate the change in the Skyhook's *specific* angular momentum
    caused by transferring a payload with a given tether-side velocity
    change (delta_v_tether), using conservation of momentum between the
    tether and the payload.

    Conservation of linear momentum (simplified, tangential exchange):

        m_payload * dv_payload = -m_tether * dv_tether

    We express the tether's momentum change in *specific* terms (per unit
    tether mass) by relating it to the angular momentum reservoir:

        delta_L_tether = r_tether * dv_tether

    For simplicity in this reduced-order model, we directly scale the
    angular momentum change by the payload-to-tether mass ratio and a
    characteristic tether radius (Earth radius + altitude is used by the
    caller). This function returns the *fractional* contribution that the
    caller scales by the tether's current orbital radius.

    Args:
        payload_mass (float): Mass of the payload (kg).
        tether_mass (float): Mass of the tether system (kg).
        delta_v_tether (float): Velocity change imparted to the tether's
            effective center of mass due to the exchange (m/s). Positive
            values represent a velocity change in the direction that
            would correspond to outgoing-payload-style momentum loss.

    Returns:
        float: delta specific angular momentum (m^2/s) -- the amount to
        SUBTRACT from the tether's angular momentum reservoir for an
        outgoing transfer (or ADD for an incoming transfer, with sign
        flipped by the caller).
    """
    mass_ratio = payload_mass / max(tether_mass, 1.0)
    return mass_ratio * delta_v_tether


def apply_payload_event(state: SkyhookState, payload_mass, delta_v_tether,
                          direction="outgoing"):
    """
    Apply a single payload capture/release event to the Skyhook's
    momentum battery, then update the resulting altitude.

    Args:
        state (SkyhookState): The Skyhook's current state (mutated in
            place).
        payload_mass (float): Mass of the payload involved (kg).
        delta_v_tether (float): Magnitude of the tether-side velocity
            change associated with this event (m/s). Always positive;
            direction determines the sign of the momentum change.
        direction (str): "outgoing" (payload launched away, tether loses
            momentum) or "incoming" (payload captured/lowered, tether
            gains momentum).

    Returns:
        float: The signed change in specific angular momentum applied
        (m^2/s). Negative for outgoing, positive for incoming.
    """
    # Characteristic radius used to convert the tether's linear velocity
    # change into a specific-angular-momentum change.
    r_tether = EARTH_RADIUS + state.altitude

    raw_delta = payload_specific_angular_momentum_delta(
        payload_mass, state.tether_mass, delta_v_tether
    ) * r_tether

    if direction == "outgoing":
        delta_L = -abs(raw_delta)
    elif direction == "incoming":
        delta_L = abs(raw_delta)
    else:
        raise ValueError(f"Unknown direction: {direction!r}. "
                          "Expected 'outgoing' or 'incoming'.")

    state.angular_momentum += delta_L
    state.update_altitude_from_angular_momentum()

    return delta_L
