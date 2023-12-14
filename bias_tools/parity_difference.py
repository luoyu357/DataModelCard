# refers to the difference in the rates of positive predictions between different groups.
import random

import pandas as pd

from decision_tree.decision_tree_model import decision_tree_l1


def calculate_parity_difference(data, protected_attribute, prediction_column):
    groups = data[protected_attribute].unique()
    print(groups)

    parity_difference_dict = {}

    for i, group_a in enumerate(groups):
        for j, group_b in enumerate(groups):
            if i < j:
                prob_a = data[data[protected_attribute] == group_a][prediction_column].mean()
                prob_b = data[data[protected_attribute] == group_b][prediction_column].mean()

                parity_difference_dict[f'{group_a} vs {group_b}'] = prob_a - prob_b

    return parity_difference_dict


# Simulated dataset with predictions
gender = []
prediction = []
actual = []
for i in range(400):
    gender.append(['F', 'M'][random.randint(0, 1)])
    prediction.append([0,1][random.randint(0, 1)])
    actual.append([0,1][random.randint(0, 1)])

# Simulated dataset with true labels and predictions
data = pd.DataFrame({
    'gender': gender,
    'prediction': prediction,
    'true_label': actual
})

protected_attribute = 'gender'
prediction_column = 'prediction'

TP = 0
FN = 0
FP = 0
TN = 0

for i in range(len(actual)):
    if actual[i] == 1:
        if prediction[i] == 1:
            TP += 1
        else:
            FN += 1
    else:
        if prediction[i] == 1:
            FP += 1
        else:
            TN += 1

score = decision_tree_l1(TP, FP, TN, FN)

result = calculate_parity_difference(data, protected_attribute, prediction_column)
print('Parity difference')
print(result)
print('decision tree score')
print(score)


TP = 0
FN = 0
FP = 0
TN = 0

for i in range(len(actual)):
    if gender[i] == 'M':
        if actual[i] == 1:
            if prediction[i] == 1:
                TP += 1
            else:
                FN += 1
        else:
            if prediction[i] == 1:
                FP += 1
            else:
                TN += 1

score_m = decision_tree_l1(TP, FP, TN, FN)

TP = 0
FN = 0
FP = 0
TN = 0

for i in range(len(actual)):
    if gender[i] == 'F':
        if actual[i] == 1:
            if prediction[i] == 1:
                TP += 1
            else:
                FN += 1
        else:
            if prediction[i] == 1:
                FP += 1
            else:
                TN += 1

score_f = decision_tree_l1(TP, FP, TN, FN)


print('decision score for Male')
print(score_m)
print('decision score for Female')
print(score_f)

# This is the absolute difference in prediction rates between groups. A parity difference close to 0 would indicate
# that the two groups are being treated similarly by the model, while a larger parity difference indicates a
# discrepancy in treatment between the groups.

