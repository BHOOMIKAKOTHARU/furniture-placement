# furniture-placement
Overview
This project aims to predict the optimal placement of furniture within a room while avoiding obstacles. The dataset was created with Furniture X and Furniture Y as target variables, which represent the coordinates of the furniture's position. The model is trained to ensure that furniture is placed in a way that avoids obstacles, considering room dimensions, furniture size, and obstacle locations.
The project includes data preprocessing, feature engineering, and model training using various machine learning techniques. Ultimately, a neural network model provided the best performance. The trained model is then deployed using Flask, and predictions can be made via a REST API using Postman.

Dataset
The dataset contains various features related to room dimensions, furniture properties, and obstacle locations. The primary objective is to predict Furniture X and Furniture Y, ensuring furniture is placed optimally while avoiding obstacles.
Features in the Dataset:
Room Dimensions: Room Width, Room Height
Furniture Properties: Furniture Width, Furniture Height
Obstacle Properties: Obstacle X, Obstacle Y, Obstacle Width, Obstacle Height
Categorical Variables (One-Hot Encoded):
Furniture Type: Chair, Table, Bed, Sofa
Obstacle Type: Wall, Pillar, Fixed Object

Data Preprocessing
Handling Missing Values:
Checked for missing values and handled them appropriately.
Exploratory Data Analysis (EDA):
Generated statistical summaries of the dataset.
Visualized distributions of numerical variables using histograms.
Created boxplots to identify outliers.
Counted occurrences of different furniture and obstacle types.

Feature Engineering:
Encoded categorical variables using one-hot encoding (pd.get_dummies).
Standardized numerical features using StandardScaler to improve model performance.

Model Training
I experimented with multiple machine learning models to predict Furniture X and Furniture Y:

1. Random Forest Regressor
Implemented RandomForestRegressor with 100 estimators.
Evaluated the model using Mean Squared Error (MSE).

2. XGBoost Regressor
Trained an XGBoost Regressor with 100 estimators and a learning rate of 0.1.
Improved performance over Random Forest but was still not optimal.

3. Neural Network (Best Performing Model)
Built a deep learning model using TensorFlow/Keras with:
Input Layer: Matching input feature size
Hidden Layers: Two fully connected layers with ReLU activation
Output Layer: Predicting two coordinates (Furniture X, Furniture Y)
Used Adam optimizer and Mean Squared Error loss function.
Achieved the best MSE, making it the final model for deployment.

API Development with Flask
To deploy the trained neural network model, we created a Flask API that accepts JSON input and returns the predicted Furniture X and Furniture Y coordinates.
API Implementation
Framework: Flask
Model Loading: joblib.load("furniture_placement_model.pkl")
Endpoint: /predict
Method: POST
Expected Input Format: JSON object containing relevant features.
Response: JSON object with predicted (Furniture X, Furniture Y) values.
Example API Request (via Postman)
POST http://127.0.0.1:5000/predict
Request Body (JSON Example):
{
  "Room Width": 10,
  "Room Height": 12,
  "Furniture Width": 2,
  "Furniture Height": 3,
  "Obstacle X": 5,
  "Obstacle Y": 6,
  "Obstacle Width": 1,
  "Obstacle Height": 2,
  "Furniture Type_Chair": 1,
  "Furniture Type_Table": 0,
  "Furniture Type_Bed": 0,
  "Furniture Type_Sofa": 0,
  "Obstacle Type_Wall": 1,
  "Obstacle Type_Pillar": 0,
  "Obstacle Type_Fixed Object": 0
}

Response:
{
  "Furniture X": 1.56,
  "Furniture Y": 3.08
}
