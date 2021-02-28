from sklearn import linear_model, preprocessing
from sklearn.model_selection import train_test_split
import pandas
import os
from flask import Flask, render_template, request

app = Flask(__name__)
regr = None


def train():
    global regr

    df = pandas.read_csv("apartmentComplexData.txt",
                         names=["IGNORED1", "IGNORED2", "complexAge", "totalRooms", "totalBedrooms",
                                "complexInhabitants", "apartmentsNr", "IGNORED8", "medianCompexValue"])
    df = df.drop(["IGNORED1", "IGNORED2", "IGNORED8"], axis=1)

    X = df.iloc[:, 0:5]
    Y = df.iloc[:, 5]

    X = preprocessing.normalize(X, norm='l2')
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    # Ordinary Least Squares
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)

    # prediction = regr.predict(X_test)
    # print("ORDINARY LEAST SQUARES")
    # print(X_test)
    # a = pandas.DataFrame({"Prediction": prediction, "Actual": y_test})
    # print(a.head())
    # print('Score: %.2f' % regr.score(X_test, y_test))
    # print("===================================")


@app.route('/')
def render_main_page():
    return render_template("index.html")


@app.route('/predict', methods=["POST"])
def predict_price():
    user_data = request.form.to_dict()
    complexAge = float(user_data["complexAge"])
    totalRooms = float(user_data["totalRooms"])
    totalBedrooms = float(user_data["totalBedrooms"])
    complexInhabitants = float(user_data["complexInhabitants"])
    apartmentsNr = float(user_data["apartmentsNr"])
    data = [[complexAge, totalRooms, totalBedrooms, complexInhabitants, apartmentsNr]]
    data = preprocessing.normalize(data, norm='l2')

    x = regr.predict(data)
    return str(x)


if __name__ == "__main__":
    train()
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
