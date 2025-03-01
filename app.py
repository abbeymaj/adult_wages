# Importing packages
from src.components.create_custom_data import CustomData
from src.components.make_prediction import MakePredictions
from flask import Flask, request, jsonify, render_template

# Instantiating the Flask app
app = Flask(__name__)

# Creating the home page
app.route('/')
def index():
    '''
    This function creates the home page for the web application.
    '''
    return render_template('index.html')

# Creating a function to predict the datapoint
app.route('/predict.html', methods=['GET', 'POST'])
def predict_datapoint():
    '''
    This function will display the prediction landing page if the method is "GET".
    If the method is "POST", the function will run the prediction.
    '''
    # Display the prediction landing page if the request method is "GET"
    if request.method == 'GET':
        return render_template('predict.html')
    else:
        # If the request method not "GET", run the prediction
        data = CustomData(
            age = int(request.form.get('age')),
            workclass = str(request.form.get('workclass')),
            education = str(request.form.get('education')),
            education_num = int(request.form.get('education_num')),
            marital_status = str(request.form.get('marital_status')),
            occupation = str(request.form.get('occupation')),
            relationship = str(request.form.get('relationship')),
            race = str(request.form.get('race')),
            sex = str(request.form.get('sex')),
            capital_gain = int(request.form.get('capital_gain')),
            capital_loss = int(request.form.get('capital_loss')),
            hours_per_week = int(request.form.get('hours_per_week')),
            native_country = str(request.form.get('native_country'))
        )
        
        # Creating a dataframe from the user entered data
        df = data.create_dataframe()
        
        # Instantiating the prediction pipeline and making predictions
        prediction = MakePredictions()
        preds = prediction.predict(df)
        
        return render_template('predict.html', results=preds, pred_df=df)

# Creating a function to return the prediction as an API call
def prediction_api():
    '''
    This function will take the data entered by the user and then 
    return the prediction as an API call.
    ---------------------
    Returns:
    ---------------------
    predictions : json - This is the prediction in a JSON format.
    '''
    # Creating the api
    if request.method == 'POST':
        data = CustomData(
            age = int(request.form.get('age')),
            workclass = str(request.form.get('workclass')),
            education = str(request.form.get('education')),
            education_num = int(request.form.get('education_num')),
            marital_status = str(request.form.get('marital_status')),
            occupation = str(request.form.get('occupation')),
            relationship = str(request.form.get('relationship')),
            race = str(request.form.get('race')),
            sex = str(request.form.get('sex')),
            capital_gain = int(request.form.get('capital_gain')),
            capital_loss = int(request.form.get('capital_loss')),
            hours_per_week = int(request.form.get('hours_per_week')),
            native_country = str(request.form.get('native_country'))
        )
        
        # Creating a dataframe from the user entered data
        df = data.create_dataframe()
        
        # Instantiating the prediction pipeline and making predictions
        prediction = MakePredictions()
        preds = prediction.predict(df)
        
        # Creating a dictionary of the predictions
        preds_dict = {
            'prediction' : preds
        }
        
        return jsonify(preds_dict)