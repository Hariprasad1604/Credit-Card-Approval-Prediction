

from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)


# Load Saved Files


model = joblib.load("model/credit_card_approval_model.pkl")
scaler = joblib.load("model/scaler.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")
binary_encoders = joblib.load("model/binary_encoders.pkl")



# Home Page


@app.route("/")
def home():
    return render_template("index.html")



# Prediction Route


@app.route("/predict", methods=["POST"])
def predict():

    try:

        

        input_data = {

            "CODE_GENDER":
                request.form["CODE_GENDER"],

            "FLAG_OWN_CAR":
                request.form["FLAG_OWN_CAR"],

            "FLAG_OWN_REALTY":
                request.form["FLAG_OWN_REALTY"],

            "CNT_CHILDREN":
                int(request.form["CNT_CHILDREN"]),

            "AMT_INCOME_TOTAL":
                float(request.form["AMT_INCOME_TOTAL"]),

            "FLAG_WORK_PHONE":
                int(request.form["FLAG_WORK_PHONE"]),

            "FLAG_PHONE":
                int(request.form["FLAG_PHONE"]),

            "FLAG_EMAIL":
                int(request.form["FLAG_EMAIL"]),

            "CNT_FAM_MEMBERS":
                float(request.form["CNT_FAM_MEMBERS"]),

            "Age":
                int(request.form["Age"]),

            "Employment_Years":
                float(request.form["Employment_Years"]),

            "NAME_INCOME_TYPE":
                request.form["NAME_INCOME_TYPE"],

            "NAME_EDUCATION_TYPE":
                request.form["NAME_EDUCATION_TYPE"],

            "NAME_FAMILY_STATUS":
                request.form["NAME_FAMILY_STATUS"],

            "NAME_HOUSING_TYPE":
                request.form["NAME_HOUSING_TYPE"],

            "OCCUPATION_TYPE":
                request.form["OCCUPATION_TYPE"]

        }

        
        # Create DataFrame
      

        user_df = pd.DataFrame([input_data])

       
        # Apply Label Encoding
       

        for column in [
            "CODE_GENDER",
            "FLAG_OWN_CAR",
            "FLAG_OWN_REALTY"
        ]:

            user_df[column] = binary_encoders[column].transform(
                user_df[column]
            )

        
        # Apply One-Hot Encoding
        

        user_df = pd.get_dummies(
            user_df,
            columns=[
                "NAME_INCOME_TYPE",
                "NAME_EDUCATION_TYPE",
                "NAME_FAMILY_STATUS",
                "NAME_HOUSING_TYPE",
                "OCCUPATION_TYPE"
            ],
            drop_first=True
        )

        
        # Match Training Features
        

        user_df = user_df.reindex(
            columns=feature_columns,
            fill_value=0
        )

        
        # Scale Features
       

        user_scaled = scaler.transform(
            user_df
        )
         
        
       # Make Prediction
        

        prediction = model.predict(user_scaled)[0]

        probability = model.predict_proba(user_scaled)[0]

        if prediction == 0:

              result = "✅ Credit Card Approved"

              confidence = round(probability[0] * 100, 2)

        else:

            result = "❌ Credit Card Rejected"

            confidence = round(probability[1] * 100, 2)

        return render_template(

            "index.html",

            prediction=result,

            confidence=confidence

        )

    except Exception as e:

        return render_template(

            "index.html",

            prediction=f"Error : {str(e)}"

        )



# Run Flask App


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)