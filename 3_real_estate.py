import numpy as np
from sklearn.linear_model import LogisticRegression

# create input dataset (x1 = Listing Price in $100k, x2 = Square Footage in thousands)
X = np.array([
    [3.5, 1.2],
    [2.0, 0.9],
    [4.2, 1.5],
    [1.8, 0.8],
    [5.5, 2.0],
    [2.5, 1.0],
    [6.0, 2.4],
    [1.5, 0.7],
    [4.8, 1.8],
    [3.0, 1.1],
])

# create output dataset (0 = Sold slower than 30 days, 1 = Sold within 30 days)
y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

# create model
model = LogisticRegression()

# train model
model.fit(X, y)

# create variable that has data to predict
new_house = np.array([[4.0, 1.4]])

# prediction
prediction = model.predict(new_house)
print("prediction about sale of new house", prediction)

probability = model.predict_proba(new_house)
print("probability of sale = ", probability)

# probability in percentage
prob_slow_pct = probability[0][0] * 100
prob_fast_pct = probability[0][1] * 100

print(f"probability of taking longer than 30 days = {prob_slow_pct:.2f}%")
print(f"probability of selling within 30 days = {prob_fast_pct:.2f}%")