from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import numpy as np

app = Flask(_name_)

# Load and preprocess the dataset
csv_path = "C:\\Users\\sreej\\OneDrive\\Documents\\Water_Quality\\water_dataX.csv"  # Update with the correct path

def load_data():
    try:
        return pd.read_csv(csv_path, encoding='ISO-8859-1')
    except UnicodeDecodeError:
        return pd.read_csv(csv_path, encoding='latin1')

def prepare_data():
    water_data = load_data()
    water_data.replace(' ', np.nan, inplace=True)

    # Define columns required for the model
    required_columns = [
        "Temp", "D.O. (mg/l)", "PH", "CONDUCTIVITY (µmhos/cm)", "B.O.D. (mg/l)",
        "NITRATENAN N+ NITRITENANN (mg/l)", "FECAL COLIFORM (MPN/100ml)",
        "TOTAL COLIFORM (MPN/100ml)Mean", "year"
    ]

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    water_data[required_columns] = imputer.fit_transform(water_data[required_columns])

    # Create pollution labels
    pollution_threshold = 500
    ph_value = 8.5
    DO = 5
    BOD = 3
    Nitrate = 10

    water_data['Polluted'] = ((water_data['CONDUCTIVITY (µmhos/cm)'] > pollution_threshold) &
                              (water_data['PH'] > ph_value) &
                              (water_data["D.O. (mg/l)"] > DO) &
                              (water_data["B.O.D. (mg/l)"] > BOD) &
                              (water_data["NITRATENAN N+ NITRITENANN (mg/l)"] > Nitrate)).astype(int)

    # Prepare features and labels
    X = water_data[required_columns]
    y = water_data['Polluted']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)

    return model, scaler, water_data

# Load model, scaler, and data
model, scaler, water_data = prepare_data()

@app.route('/')
def home():
    # Retrieve unique states and locations for autocomplete suggestions
    states = water_data['STATE'].dropna().unique().tolist()
    locations = water_data['LOCATIONS'].dropna().unique().tolist()
    return render_template('home.html', result=None, states=states, locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    state = request.form['state']
    location = request.form['location']

    # Filter data by selected state and location
    data_point = water_data[(water_data['STATE'].str.lower() == state.lower()) & 
                            (water_data['LOCATIONS'].str.contains(location, case=False, na=False))]

    if data_point.empty:
        result = "No data available for the selected location."
        return render_template('home.html', result=result, states=water_data['STATE'].unique(), locations=water_data['LOCATIONS'].unique())
    else:
        # Prepare features for prediction
        features = data_point.drop(columns=['Polluted', 'STATE', 'LOCATIONS', 'STATION CODE'], errors='ignore')
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        result = "Polluted" if prediction == 1 else "Not Polluted"

    return render_template('home.html', result=f"{location} is: {result}", states=water_data['STATE'].unique(), locations=water_data['LOCATIONS'].unique())

if _name_ == '_main_':
    app.run(debug=True)