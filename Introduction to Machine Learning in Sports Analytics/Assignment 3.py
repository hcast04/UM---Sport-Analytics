import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

observations=pd.read_csv("assets/observations.csv", index_col=0)

# This is the tricky part - dropping the Golden Knights using away_cap and home_cap
observations=observations.dropna(subset=["away_cap","home_cap"])

train=observations[0:800]
validate=observations[800:]

X_train=train[train.columns[0:-1]]
y_train=train[train.columns[-1]]

X_validate=validate[validate.columns[0:-1]]
y_validate=validate[validate.columns[-1]]


parameters={'max_depth':(3,4,5,6,7,8,9,10), 
            'min_samples_leaf':(1,5,10,15,20,25)}

clf=GridSearchCV(estimator=DecisionTreeClassifier(random_state=1337), param_grid=parameters, cv=10, scoring='accuracy')
clf.fit(X_train,y_train).score(X_validate,y_validate)


print(clf.score(X_validate,y_validate))
print(clf.best_params_)