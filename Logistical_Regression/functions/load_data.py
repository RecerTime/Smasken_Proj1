import pandas as pd
import os

def load_data():
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../training_data_vt2025.csv"))
    data = pd.read_csv(csv_path)
    return data