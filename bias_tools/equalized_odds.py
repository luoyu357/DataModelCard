# Equalized odds requires that the classifier has equal true positive rates and equal false positive rates for each
# group. In the case of multi-class classification, this concept extends to each class individually.

import pandas as pd


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
data = pd.DataFrame({
    'gender': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'] * 10,
    'prediction': [0, 1, 2, 0, 1, 2, 0, 1, 2, 0] * 10,
    'true_label': [0, 1, 2, 0, 0, 1, 2, 2, 1, 0] * 10
})

protected_attribute = 'gender'
prediction_column = 'prediction'
true_label_column = 'true_label'

result = calculate_equalized_odds(data, protected_attribute, prediction_column, true_label_column)
print(result)

# It means that for a fair model, the true positive rate and the false positive rate should be equal for groups
# a and # b (or for all groups considered).
