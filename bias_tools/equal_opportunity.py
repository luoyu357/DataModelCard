# requires the true positive rate (TPR) to be equal across different groups, meaning that all groups have an equal
# opportunity to be correctly identified as positive.
import random

import pandas as pd

from decision_tree.decision_tree_model import decision_tree_l1

# true positive / actual positive for each class for each group
def calculate_equal_opportunity(data, protected_attribute, prediction_column, true_label_column):
    groups = data[protected_attribute].unique()
    classes = data[prediction_column].unique()

    equal_opportunity_dict = {}

    for c in classes:
        class_dict = {}
        for group in groups:
            group_data = data[data[protected_attribute] == group]
            true_positives = sum((group_data[prediction_column] == c) & (group_data[true_label_column] == c))

            actual_positives = sum(group_data[true_label_column] == c)

            tpr = true_positives / actual_positives if actual_positives != 0 else 0
            class_dict[group] = tpr

        equal_opportunity_dict[c] = class_dict

    return equal_opportunity_dict


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
true_label_column = 'true_label'

result = calculate_equal_opportunity(data, protected_attribute, prediction_column, true_label_column)
print('Equal Opportunity')
print(result)


for j in range(2):
    TP = 0
    FN = 0
    FP = 0
    TN = 0

    for i in range(len(actual)):
        if gender[i] == 'M':
            if actual[i] == j:
                if prediction[i] == j:
                    TP += 1
                else:
                    FN += 1
            else:
                if prediction[i] == j:
                    FP += 1
                else:
                    TN += 1

    score_m = decision_tree_l1(TP, FP, TN, FN)

    print('decision score for M with '+str(j))
    print(score_m)

for j in range(2):
    TP = 0
    FN = 0
    FP = 0
    TN = 0

    for i in range(len(actual)):
        if gender[i] == 'F':
            if actual[i] == j:
                if prediction[i] == j:
                    TP += 1
                else:
                    FN += 1
            else:
                if prediction[i] == j:
                    FP += 1
                else:
                    TN += 1

    score_f = decision_tree_l1(TP, FP, TN, FN)
    print('decision score for F with ' + str(j))
    print(score_f)





# It has the same true positive rate (also called sensitivity or recall), meaning that the rate of correctly
# predicted positive instances is equal across different groups.
