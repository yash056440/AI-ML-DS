import numpy as np
from sklearn.linear_model import LogisticRegression

# create input dataset (x1 = Time on Site in minutes, x2 = Pages Viewed)
X = np.array([
    [2.5, 3],
    [12.0, 15],
    [4.0, 5],
    [15.5, 18],
    [3.2, 4],
    [10.0, 12],
    [6.5, 7],
    [18.0, 20],
    [5.0, 6],
    [13.5, 16],
])

# create output dataset (0 = Abandoned Cart, 1 = Completed Purchase)
y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

# create model
model = LogisticRegression()

# train model
model.fit(X, y)

# create variable that has data to predict
new_shopper = np.array([[2, 11]])

# prediction
prediction = model.predict(new_shopper)
print("prediction about purchase of new shopper", prediction)

probability = model.predict_proba(new_shopper)
print("probability of purchase = ", probability)