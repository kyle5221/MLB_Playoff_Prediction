import numpy as np
import matplotlib.pyplot as plt  
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import pickle

df = pd.read_csv('mlb_elo.csv')

df['TeamYear'] = df['season'].astype(str) + '-' + df['team1']
df_stage = df[df['playoff']==1]
        
playoffteams = df_stage['TeamYear'].unique().tolist()
allteams = df['TeamYear'].unique().tolist()
labels = list()

for team in allteams:
    if team in playoffteams:
        labels.append(1)
    else:
        labels.append(0)
        
teamlabels = pd.DataFrame(
    {'TeamYear': allteams,
     'Playoff': labels
         })

datacolumns = ["M1","M2","M3","M4","M5","M6","M7","M8","M9","M10","M11","M12","M13","M14","M15","M16","M17","M18","M19","M20","score1","score2"]
for column in datacolumns:
    teamlabels[column] = np.nan
        
        
teamlabels = teamlabels.set_index('TeamYear')

for team in allteams:
    teamrecord = df[(df['TeamYear']==team) & (df['playoff']==0)]
    teamrecord = teamrecord[['result','score1','score2']]
    teamrecord = teamrecord.head(20)
    teamrecord.at['score1','result'] = teamrecord['score1'].sum()
    teamrecord.at['score2','result'] = teamrecord['score2'].sum()
    teamrecord = teamrecord[['result']]
    teamrecord = teamrecord.T
    teamrecord = teamrecord.rename(index={'result': team})
    teamrecord.columns = datacolumns
    teamlabels.update(teamrecord)
    lst = [teamrecord]
    del lst

y = teamlabels[['Playoff']]
teamlabels = teamlabels.drop(['Playoff'], axis=1)
X = teamlabels.copy()
X_pts = X[['score1','score2']]
X = X.drop(['score1','score2'],axis=1)
X = pd.get_dummies(X)
X['Win']=X['M1_W']+X['M2_W']+X['M3_W']+X['M4_W']+X['M5_W']+X['M6_W']+X['M7_W']+X['M8_W']+X['M9_W']+X['M10_W']+X['M11_W']+X['M12_W']+X['M13_W']+X['M14_W']+X['M15_W']+X['M16_W']+X['M17_W']+X['M18_W']+X['M19_W']+X['M20_W']
X['Lost']=X['M1_L']+X['M2_L']+X['M3_L']+X['M4_L']+X['M5_L']+X['M6_L']+X['M7_L']+X['M8_L']+X['M9_L']+X['M10_L']+X['M11_L']+X['M12_L']+X['M13_L']+X['M14_L']+X['M15_L']+X['M16_L']+X['M17_L']+X['M18_L']+X['M19_L']+X['M20_L']
X = X[['Win','Lost']]
X['score1'] = X_pts['score1']
X['score2'] = X_pts['score2']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
rnd_clf = RandomForestClassifier(n_estimators=3000, max_leaf_nodes=16)

rnd_clf.fit(X_train, y_train.values.ravel())
pickle.dump(rnd_clf, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
print(model.predict([[20, 0, 1000, 0]]))