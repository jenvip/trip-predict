import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

print("1. Loading the massive CSV file...")
df = pd.read_csv('nj_transit_data.csv')

#shrink the data by taking 50000rows 
df_small = df.sample(n=50000, random_state=42)

# Save this small CSV so you can upload it to GitHub for others to see
df_small.to_csv('nj_transit_data_small.csv', index=False)
print(f"Data shrunk from {len(df)} rows to {len(df_small)} rows.")

#prepare data
print("2. Cleaning and preparing data...")
relevant_cols = ['date', 'line', 'stop_sequence', 'from', 'to', 'scheduled_time', 'delay_minutes']
df_clean = df_small[relevant_cols].copy().dropna()

#handle time
df_clean['scheduled_dt'] = pd.to_datetime(df_clean['scheduled_time'])
df_clean['sch_min'] = df_clean['scheduled_dt'].dt.hour * 60 + df_clean['scheduled_dt'].dt.minute
df_clean['day_of_week'] = pd.to_datetime(df_clean['date']).dt.dayofweek

#encode
le = LabelEncoder()
for col in ['line', 'from', 'to']:
    df_clean[col] = le.fit_transform(df_clean[col].astype(str))

#train a "lite" model
print("3. Training a lightweight model...")
X = df_clean[['line', 'stop_sequence', 'from', 'to', 'sch_min', 'day_of_week']]
y = df_clean['delay_minutes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#max_depth=10 keeps the file size SMALL. 
#compress=3 shrinks the file significantly.
rf_model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

#save the encoder too
joblib.dump(le, 'label_encoder.joblib')

#save as .pkl
print("4. Saving the model to 'njt_delay_model.pkl'...")
joblib.dump(rf_model, 'njt_delay_model.pkl', compress=3)

print("Success! Created 'nj_transit_data_small.csv' and 'njt_delay_model.pkl'")