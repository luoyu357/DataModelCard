# Equalized odds requires that the classifier has equal true positive rates and equal false positive rates for each
# group. In the case of multi-class classification, this concept extends to each class individually.
import random

import pandas as pd

from decision_tree.decision_tree_model import decision_tree_l1

#
def calculate_equalized_odds(data, protected_attribute, prediction_column, true_label_column):
    groups = data[protected_attribute].unique()
    classes = data[prediction_column].unique()

    equalized_odds_dict = {}

    for group in groups:
        group_data = data[data[protected_attribute] == group]
        group_odds = {}

        for c in classes:
            true_positives = sum((group_data[prediction_column] == c) & (group_data[true_label_column] == c))
            false_positives = sum((group_data[prediction_column] == c) & (group_data[true_label_column] != c))
            actual_positives = sum(group_data[true_label_column] == c)
            actual_negatives = sum(group_data[true_label_column] != c)

            tpr = true_positives / actual_positives if actual_positives != 0 else 0
            fpr = false_positives / actual_negatives if actual_negatives != 0 else 0

            group_odds[c] = {'TPR': tpr, 'FPR': fpr}

        equalized_odds_dict[group] = group_odds

    return equalized_odds_dict


# Simulated dataset with true labels and predictions
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

result = calculate_equalized_odds(data, protected_attribute, prediction_column, true_label_column)
print('Equalized odds')
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

# It means that for a fair model, the true positive rate and the false positive rate should be equal for groups
# a and # b (or for all groups considered).
