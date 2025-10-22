---

layout: page
title: Electricity Production Optimizer
description: Optimizing electricity production from thermal and hydro plants
img: assets/img/optimizer.png
lang: en
importance: 3
category: project
-----------------

## Introduction

The **Electricity Production Optimizer** is an interactive web application developed with **Streamlit** and **Gurobi MILP**.
It allows the user to plan and optimize electricity production across multiple periods from **thermal and hydroelectric plants** while minimizing operational costs and respecting physical constraints.

The goal is to provide energy planners and analysts with a flexible tool to test production scenarios, visualize results, and export detailed reports for decision-making.

---

## Optimization Strategy

The application uses a **Mixed-Integer Linear Programming (MILP)** model to determine the optimal generation schedule, considering:

* **Thermal plants**: minimum and maximum generation, start-up costs, variable costs
* **Hydro plants**: reservoir volumes, production limits, and pumping options
* **Hourly demand**: the optimization ensures that total generation meets the demand per period
* **Optional cyclic planning**: consider 24h operation cycles for daily optimization

Users can adjust parameters through **interactive sliders** or upload **JSON files** for demand and plant configurations.

---

## MILP Formulation

The model minimizes total production cost:

$$
\min \sum_{t=1}^{T} \sum_{p=1}^{P} \left[ (P_{t,p} - N_{t,p} \cdot P^{\min}_t) \cdot C^{MWh}_t \cdot H_p + N_{t,p} \cdot C^{base}_t \cdot H_p + X_{t,p} \cdot C^{start}_t \right]
$$

subject to:

* Demand satisfaction:

$$
\sum_{i \in \text{thermal}} p_{i,t} + \sum_{j \in \text{hydro}} p_{j,t} = D_t \quad \forall t
$$

* Thermal generation limits:

$$
p_{i,\min} \cdot u_{i,t} \le p_{i,t} \le p_{i,\max} \cdot u_{i,t}
$$

* Hydro reservoir constraints:

$$
V_{j,t+1} = V_{j,t} - p_{j,t} + P_{j,t}^{\text{pump}}
$$

* Binary variables for start-up and shut-down decisions for thermal plants

* Optional constraints for pumping and cyclic operation

---

## Methodology

* Optimization is performed **hourly for 24h periods**
* Users can **adjust plant parameters**, including:

  * number of units, min/max generation, base and variable costs, start-up costs, reservoir volumes
* Users can **upload JSON files** with their own demand and plant data
* Results include: production schedule, plant activation, costs, and metrics
* Export results as **Excel** with styled tables and key metrics

---

## JSON File Format

### Demand file (`demand_config.json`)

```json
{
  "periods": [
    {"name": "0h-6h", "hours": 6, "demand_mw": 15000},
    {"name": "6h-9h", "hours": 3, "demand_mw": 30000}
  ]
}
```

* `name`: name of the period
* `hours`: duration of the period in hours
* `demand_mw`: electricity demand in MW

### Plants file (`plants_config.json`)

```json
{
  "thermal_plants": [
    {
      "id": "A",
      "name": "Type A",
      "default": {"amount": 12, "p_min": 850, "p_max": 2000, "c_base": 1000, "c_mwh": 2.0, "c_start": 2000},
      "ranges": {"amount": [1, 20], "p_min": [100, 2000, 50], "p_max": [500, 5000, 50],
                 "c_base": [0, 5000, 50], "c_mwh": [0.0, 5.0, 0.1], "c_start": [0, 3000, 50]}
    }
  ],
  "hydro_plants": [
    {
      "id": "H1",
      "name": "Hydro 1",
      "default": {"amount": 1, "p_min": 900, "p_max": 900, "c_base": 90, "c_mwh": 0, "c_start": 1500, "v_abai": 0.31},
      "ranges": {"amount": [0, 2], "p_min": [500, 2000, 50], "p_max": [500, 2000, 50],
                 "c_base": [0, 500, 10], "c_start": [0, 2000, 50], "v_abai": [0.0, 1.0, 0.01]}
    }
  ]
}
```

* `thermal_plants` and `hydro_plants` are lists of plants
* Each plant contains: `id`, `name`, `default`, `ranges`
* `default`: default values for optimization
* `ranges`: min/max values and step for sliders

---

## Metrics Reported

* Total production cost (€)
* Optimization time (s)
* Optimality gap (%)
* Number of variables and constraints
* Nodes explored
* Model name

Results can be exported to **Excel** with styling for readability.

---

## Code Structure

```text
electricity_optimizer/
│
├─ app.py # Main Streamlit application
├─ demand_config.json # Optional demand configuration
├─ plants_config.json # Optional plants configuration
├─ requirements.txt # Python dependencies
└─ README.md # This file
```

---

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/yourusername/electricity-optimizer.git
cd electricity-optimizer
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
streamlit run app.py
```

---

## Developer

**José Eduardo** - [Personal Website](https://jeduapf.github.io)

---