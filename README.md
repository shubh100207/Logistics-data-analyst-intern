# Strategic Planning and Data Exploration in Logistics (Week 1)

## Project Overview
This repository contains the Week 1 technical implementation for strategic last-mile logistics optimization at **FastTrack Logistics**. The goal is to apply predictive analytics, machine learning, and combinatorial optimization to enhance supply chain efficiency and reduce costs.

## Key Performance Indicators (KPIs)
- **On-Time In-Full (OTIF) Rate:** Target $\ge 96\%$ (Baseline: $84\%$)
- **Cost Per Delivery (CPD):** Target $< \$6.90$ (Baseline: $\$8.45$)
- **Inventory Turnover Ratio (ITR):** Target $9.5\times$ (Baseline: $6.2\times$)

## Data Science Methodologies
1. **Demand Forecasting:** Random Forest Regression for SKU-level daily order predictions.
2. **Geospatial Clustering:** K-Means clustering to place Micro-Fulfillment Centers (MFCs) dynamically.
3. **Vehicle Routing (VRP):** Constraint programming solver framework to minimize delivery mileage and transit times.

## How to Run
```bash
pip install -r requirements.txt
python main.py
