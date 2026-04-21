import pandas as pd
import numpy as np
import json
import os
import random
from tqdm import tqdm

class GridSimulator:
    def __init__(self, input_csv):
        print(f"Initializing GridGuard Simulation Engine for: {input_csv}")
        self.df = pd.read_csv(input_csv)
        self.output_path = "../../data/grid_simulated_dataset.csv"
        
        # Region definitions (TRNC Localization)
        self.regions = {
            "North": {"lat_range": (35.3, 35.4), "lon_range": (33.2, 33.5), "feeders": 5},
            "South": {"lat_range": (35.1, 35.2), "lon_range": (33.2, 33.4), "feeders": 8},
            "Urban": {"lat_range": (35.15, 35.25), "lon_range": (33.3, 33.4), "feeders": 12},
            "Rural": {"lat_range": (35.0, 35.1), "lon_range": (33.8, 34.2), "feeders": 4}
        }

    def simulate_grid_context(self):
        """Generates realistic grid load indices correlating with consumption."""
        print(">> Generating Grid Load Correlation...")
        num_days = self.df.shape[1] - 2 # Exclude CONS_NO and FLAG
        
        # Base daily load curve (Peak at evening, trough at night)
        # Using a simple sine wave for simulation of 24h cycle
        t = np.linspace(0, 2*np.pi, 24)
        base_curve = 0.5 + 0.3 * np.sin(t - np.pi/2) # Peak at 6PM (18:00)
        
        # Extend to full duration of dataset
        grid_load = np.tile(base_curve, (num_days // 24) + 1)[:num_days]
        return grid_load

    def run_transformation(self):
        print("=" * 60)
        print("  STARTING SGCC DATASET UPGRADE (GRID-GRADE SIMULATION)")
        print("=" * 60)

        # 1. Base Preservation
        cons_no = self.df['CONS_NO'].values
        labels = self.df['FLAG'].values
        consumption_matrix = self.df.drop(['CONS_NO', 'FLAG'], axis=1).values
        
        processed_records = []
        
        # 2. Geospatial & Grid Clustering
        print(">> Building Grid Topology (Regions, Feeders, Transformers)...")
        
        # Pre-assign households to a static region and topology
        unique_households = list(set(cons_no))
        household_map = {}
        
        for hid in unique_households:
            region_name = random.choice(list(self.regions.keys()))
            region = self.regions[region_name]
            
            feeder_id = f"{region_name[:1]}-F{random.randint(1, region['feeders'])}"
            transformer_id = f"TX-{random.randint(100, 999)}"
            
            # Synthetic but consistent Lat/Lon
            lat = random.uniform(*region['lat_range'])
            lon = random.uniform(*region['lon_range'])
            
            cust_type = "Residential" if random.random() > 0.15 else "Commercial"
            
            household_map[hid] = {
                "region": region_name,
                "feeder": feeder_id,
                "transformer": transformer_id,
                "lat": round(lat, 5),
                "lon": round(lon, 5),
                "type": cust_type
            }

        # 3. Time-Series Flattening & Enrichment
        print(">> Flattening Time-Series & Injecting Grid Context...")
        
        grid_load = self.simulate_grid_context()
        
        # We'll process a representative subset (e.g., first 500 households) 
        # to ensure the simulator is fast for the real-time API
        subset_size = 500 
        
        final_data = []
        
        for i in tqdm(range(min(subset_size, len(cons_no)))):
            hid = cons_no[i]
            meta = household_map[hid]
            is_theft = labels[i]
            
            # Determine Anomaly Type
            anomaly_type = "None"
            if is_theft:
                # Heuristic categorization based on mean consumption
                avg_cons = np.mean(consumption_matrix[i])
                if avg_cons < 0.5: anomaly_type = "Meter Bypass"
                elif avg_cons < 2.0: anomaly_type = "Constant Reduction"
                else: anomaly_type = "Partial Suppression"

            for day in range(min(30, consumption_matrix.shape[1])): # Limit to 30 days for simulation
                record = {
                    "timestamp": f"2026-04-{day+1:02d}T12:00:00Z",
                    "household_id": str(hid),
                    "consumption_kwh": round(float(consumption_matrix[i][day]), 3),
                    "grid_load_index": round(float(grid_load[day % len(grid_load)]), 3),
                    "region_id": meta["region"],
                    "feeder_id": meta["feeder"],
                    "transformer_id": meta["transformer"],
                    "lat": meta["lat"],
                    "lon": meta["lon"],
                    "customer_type": meta["type"],
                    "anomaly_label": int(is_theft),
                    "anomaly_type": anomaly_type
                }
                final_data.append(record)

        # 4. Save
        output_df = pd.DataFrame(final_data)
        os.makedirs("../../data", exist_ok=True)
        output_df.to_csv(self.output_path, index=False)
        
        print("=" * 60)
        print(f"[SUCCESS] Grid-Grade Dataset Generated: {self.output_path}")
        print(f"Total Simulated Events: {len(output_df):,}")
        print("=" * 60)

if __name__ == "__main__":
    # Ensure relative paths work
    input_file = "../../data/data_set_cleaned.csv"
    if not os.path.exists(input_file):
        input_file = "../data/data_set_cleaned.csv"
        
    simulator = GridSimulator(input_file)
    simulator.run_transformation()
