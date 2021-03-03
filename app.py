from flask import Flask
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

filename = 'finalmodel.pkl'
classifier = pickle.load(open(filename, 'rb'))
df = pd.read_csv("teams.csv")
ohe = OneHotEncoder(sparse=False)
ohe.fit_transform(df[['team1','team2']])

result={'Sunrisers Hyderabad': 0,
 'Mumbai Indians': 1,
 'Royal Challengers Bangalore': 2,
 'Kolkata Knight Riders': 3,
 'Chennai Super Kings': 4,
 'Delhi Daredevils': 5,
 'Rajasthan Royals': 6,
 'Kings XI Punjab': 7}

app = Flask(__name__)

@app.route('/predict/<msg>',methods=['GET','POST'])
def predict(msg):
    l=msg.split(',')
    user_team = [[l[0], l[1]]]
    team = ohe.transform(user_team)
    team = np.column_stack([team, np.array([int(l[2])])])
    team_day = np.array(team).reshape((1, -1))
    return list(result.keys())[list(result.values()).index(classifier.predict(team_day).tolist()[0])]


if __name__ == '__main__':
    app.run(debug=True)