import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

# Create flask app
flask_app = Flask(__name__)
with open("diploma_web\web\models\model_cat.pkl", "rb") as pkl_file: 
    model = pickle.load(pkl_file)

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
       
    form_data = request.form.to_dict()
    # Initialize DataFrame
    df = pd.DataFrame(columns=list(form_data.keys()))
    df = pd.DataFrame([form_data], columns=list(form_data.keys()))
    # Convert to proper types
    df = df.astype({'baths':float, 'fireplace':bool, 'sqft':float, 'zipcode':str, 'state':str, 
       'private_pool':bool, 'status_new':str, 'property_type':str, 'city':str, 'year built':str,
       'heating':bool, 'cooling':bool, 'parking':bool, 'remodeled':bool, 'schools_high_rating':bool,
       'mean_distance_school':str, 'school_info':bool})
    

    prediction = model.predict(df)
    

    return render_template("index.html", prediction_text = "The price of real estate is {}".format(np.expm1(prediction)))

if __name__ == "__main__":
    flask_app.run(debug=True)

