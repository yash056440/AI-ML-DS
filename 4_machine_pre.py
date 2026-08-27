import numpy as np
from sklearn.linear_model import LogisticRegression

# create input dataset (x1 = Operating Temperature in °C, x2 = Vibration Level)
X = np.array([
    [65, 2.1],
    [95, 6.8],
    [70, 2.5],
    [102, 7.5],
    [68, 2.3],
    [98, 7.0],
    [72, 2.8],
    [110, 8.2],
    [66, 2.0],
    [90, 6.5],
])

# create output dataset (0 = Operate Normally, 1 = Fail)
y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

# create model
model = LogisticRegression()

# train model
model.fit(X, y)

# create variable that has data to predict
new_machine = np.array([[85, 5.5]])

# prediction
prediction = model.predict(new_machine)
print("prediction about machine failure", prediction)

probability = model.predict_proba(new_machine)
print("probability of failure = ", probability)

# probability in percentage
prob_normal_pct = probability[0][0] * 100
prob_fail_pct = probability[0][1] * 100

print(f"probability of operating normally = {prob_normal_pct:.2f}%")
print(f"probability of failure = {prob_fail_pct:.2f}%")