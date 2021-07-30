from flask import Flask,render_template, request, redirect
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
from io import BytesIO
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/plot', methods =["GET", "POST"])
def salary():
    if request.method == "POST":
        mycsv = request.form.get("mycsv")
        eliminate = request.form.get("eliminate")
        eliminated = eliminate.split(',')
        target = request.form.get("target")
        eliminated.append(target)
    

        img = BytesIO()
        df = pd.read_csv(mycsv)

        X = df.drop(eliminated,axis=1)
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

        lm = LinearRegression()
        lm.fit(X_train,y_train)

        predictions = lm.predict(X_test)
        plt.scatter(y_test,predictions)
        plt.xlabel('y_test')
        plt.ylabel('My_predictions')
        plt.title('Scatter Plot')

        score = lm.score(X_test,y_test)
        score = score*100
        score = str(score)

        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    return render_template("plot.html", plot_url = plot_url,accuracy=score)
    
app.run(debug=True)