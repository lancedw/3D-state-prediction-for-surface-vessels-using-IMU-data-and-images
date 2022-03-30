import json
import pandas as pd
from datetime import datetime
logs = pd.read_csv("Work-log_Thesis.csv")
logs["Start"] = pd.to_datetime(logs["Start"], format='%H:%M', errors='ignore')
logs["End"] = pd.to_datetime(logs["End"], format='%H:%M', errors='ignore')

logs["diff"] =  logs["End"] - logs["Start"]

total = logs["diff"].sum()

print(round(total.total_seconds()/3600, 2))


