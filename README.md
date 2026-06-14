# Skyhook Momentum Management System

An educational simulator for an orbital Skyhook (rotating momentum-exchange
tether), demonstrating momentum transfer, orbital decay, and long-term
"momentum battery" management.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Modes

### Mode 1 — Physics Demonstration (Simulator V1)
Runs a sequence of outgoing-only payload launches and shows cumulative
momentum loss and orbital altitude decay. Answers: *"Why is momentum
management necessary?"*

### Mode 2 — Momentum Management Simulator (Simulator V2)
Runs a long-term (30–1000 day) simulation balancing outgoing launches
against incoming returns. Tracks:

- Momentum battery level (%) — 100% = nominal, 0% = depleted
- Orbital altitude evolution
- Daily momentum deficit vs. thruster correction
- Cumulative thruster fuel usage

Traffic balance examples:
- 10 launches/day, 2 returns/day → battery drains over time
- 10 launches/day, 10 returns/day → battery stays roughly stable
- 10 launches/day, 15 returns/day → battery charges up

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit dashboard (both modes) |
| `simulation_engine.py` | Mode 1 and Mode 2 simulation loops |
| `momentum_model.py` | Core physics: SkyhookState, momentum battery, orbit-altitude relation |
| `requirements.txt` | Python dependencies |

## Physics Model Notes

This is a **reduced-order educational model**, not a research-grade
simulator:

- The Skyhook is treated as a point-mass with a "specific angular
  momentum reservoir" (the momentum battery).
- Outgoing payloads subtract from the reservoir; incoming payloads add
  to it, via simple conservation of momentum scaled by mass ratios.
- Orbital altitude is derived from the circular-orbit relation
  `L = sqrt(mu * r)`, so altitude responds directly and consistently to
  changes in the momentum reservoir.
- Thruster fuel usage is a simplified linear proportionality to the
  correction magnitude — sufficient to show the *trend* (more deficit →
  more fuel), not a rocket-equation-accurate figure.

## Extending the Project

- Add rotational dynamics (tether spin rate, tip velocity) to
  `momentum_model.py`.
- Replace the linear fuel model with a Tsiolkovsky rocket-equation
  estimate.
- Add stochastic/variable daily traffic instead of fixed
  launches/returns per day.
