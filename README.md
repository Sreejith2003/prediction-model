Water Quality Prediction System
This project is a web-based application for predicting water quality using machine learning. The system allows users to input the state and location to predict whether the water in that location is polluted based on historical water quality data.

Project Features:
Data Preprocessing: Loads and preprocesses water quality data, imputing missing values.
Prediction Model: Trains a Random Forest Classifier to predict water pollution based on various water quality parameters.
User Interface: A web form where users can enter a state and location, and receive a prediction on water pollution.
Responsive Web Design: The app features a modern and clean design with a gradient background and intuitive interface.
Requirements
Python 3.x
Flask - Web framework for Python.
pandas - For data manipulation and analysis.
scikit-learn - For machine learning models and preprocessing.
numpy - For numerical operations.
Installation Instructions
Follow these steps to get the project running on your local machine:

1. Clone the Repository
bash
Copy code
git clone https://github.com/yourusername/water-quality-prediction.git
cd water-quality-prediction
2. Install Dependencies
You can create a virtual environment and install the dependencies using pip:

bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows use 'venv\Scripts\activate'
pip install -r requirements.txt
Note: If you don't have a requirements.txt file yet, create one by running pip freeze > requirements.txt.

The dependencies include:

Flask
pandas
scikit-learn
numpy
3. Update Dataset Path
In the app.py file, update the path to your dataset (water_data.csv) if necessary.

python
Copy code
csv_path = "path_to_your_data_file.csv"
Make sure the dataset is structured correctly with columns like "Temp", "D.O. (mg/l)", "PH", etc.

4. Run the Application
Once the dependencies are installed and the dataset path is updated, you can start the application:

bash
Copy code
python app.py
The application will run on http://127.0.0.1:5000/ by default.

How it Works
Data Loading and Preprocessing: The dataset is loaded using pandas. Missing values are handled using SimpleImputer. Data is then cleaned and features are scaled using StandardScaler.
Model Training: A RandomForestClassifier is used to classify the water as polluted or not polluted. The model is trained using the processed data.
User Input: The user inputs the state and location, and the model predicts whether the water is polluted at that location.
Prediction Result: The result is displayed on the webpage with styling to show whether the water is "Polluted" or "Not Polluted".
Web Interface
The web interface has the following sections:

Home Page: The user can input the state and location of the water body.
Prediction Page: Displays the result of the prediction. If the water is polluted, the result is displayed in red, and if it is not polluted, it is shown in green.
Go Back: A button to return to the home page.
Example
Here’s an example of what the application looks like:

The user inputs a state and location (e.g., "California" and "Lake Tahoe").
The system processes the data and predicts whether the water at that location is polluted or not.
The prediction is displayed with a color-coded message (red for polluted, green for not polluted).
Contributing
If you’d like to contribute to this project, feel free to fork the repository, make changes, and submit a pull request. Contributions are always welcome!

License
This project is licensed under the MIT License - see the LICENSE file for details.
