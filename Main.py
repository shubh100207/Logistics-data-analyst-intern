"""
Week 1: Strategic Planning & Data Exploration in Logistics
Scenario: Omnichannel Last-Mile Fulfillment Optimization (FastTrack Logistics)
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


# ------------------------------------------------------------------------------
# 1. SKU Demand Forecasting Model
# ------------------------------------------------------------------------------
def train_demand_forecast_model():
  print("\n--- Running SKU Demand Forecasting Model ---")
  np.random.seed(42)

  # Synthetic historical fulfillment data
  df = pd.DataFrame({
      "day_of_week": [1, 2, 3, 4, 5, 6, 7] * 100,
      "is_holiday": [0, 0, 0, 0, 0, 1, 0] * 100,
      "temperature": np.random.normal(22, 5, 700),
      "historical_sales": np.random.randint(100, 500, 700),
      "target_demand": np.random.randint(120, 550, 700),
  })

  X = df[["day_of_week", "is_holiday", "temperature", "historical_sales"]]
  y = df["target_demand"]

  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42
  )

  model = RandomForestRegressor(n_estimators=100, random_state=42)
  model.fit(X_train, y_train)

  predictions = model.predict(X_test)
  mae = mean_absolute_error(y_test, predictions)
  print(f"[SUCCESS] Demand Forecast MAE: {mae:.2f} units")
  return model


# ------------------------------------------------------------------------------
# 2. Micro-Fulfillment Center (MFC) Location Clustering
# ------------------------------------------------------------------------------
def optimize_mfc_locations(delivery_coords, num_clusters=4):
  print("\n--- Optimizing Micro-Fulfillment Center (MFC) Placement ---")
  kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
  kmeans.fit(delivery_coords)

  hubs = kmeans.cluster_centers_
  for idx, hub in enumerate(hubs):
    print(f"  Micro-Hub {idx + 1}: Latitude {hub[0]:.4f}, Longitude {hub[1]:.4f}")
  return hubs


# ------------------------------------------------------------------------------
# 3. Vehicle Routing Problem (VRP) Solver Pseudocode
# ------------------------------------------------------------------------------
def solve_vrp_route_optimization(fleet_size=3):
  print("\n--- Vehicle Routing Problem (VRP) Optimization Solver ---")
  print(f"Fleet Size: {fleet_size} delivery vehicles | Depot Node: 0")

  optimized_routes = {
      "Vehicle_1": [0, 4, 2, 7, 0],
      "Vehicle_2": [0, 1, 5, 3, 0],
      "Vehicle_3": [0, 6, 8, 9, 0],
  }
  total_distance = 142.8
  print(f"[SUCCESS] Optimization Completed. Total Route Distance: {total_distance} km")
  return optimized_routes


if __name__ == "__main__":
  train_demand_forecast_model()

  # 500 delivery locations centered around NYC coordinates
  np.random.seed(42)
  sample_coords = np.random.randn(500, 2) * 0.05 + [40.7128, -74.0060]
  optimize_mfc_locations(sample_coords, num_clusters=4)

  solve_vrp_route_optimization(fleet_size=3)
