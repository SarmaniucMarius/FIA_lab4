from sklearn import linear_model, preprocessing
from sklearn.model_selection import train_test_split
import pandas
import os
from flask import Flask

app = Flask(__name__)


def train():
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

    prediction = regr.predict(X_test)
    print("ORDINARY LEAST SQUARES")
    a = pandas.DataFrame({"Prediction": prediction, "Actual": y_test})
    print(a.head())
    print('Score: %.2f' % regr.score(X_test, y_test))
    print("===================================")


@app.route('/hello')
def hello_world():
    return 'Hello World!'


if __name__ == "__main__":
    train()
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
