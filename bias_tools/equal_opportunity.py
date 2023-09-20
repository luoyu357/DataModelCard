# requires the true positive rate (TPR) to be equal across different groups, meaning that all groups have an equal
# opportunity to be correctly identified as positive.


import pandas as pd


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


# Simulated dataset with true labels and predictions
data = pd.DataFrame({
    'gender': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'] * 10,
    'prediction': [0, 1, 2, 0, 1, 2, 0, 1, 2, 0] * 10,
    'true_label': [0, 1, 2, 0, 0, 1, 2, 2, 1, 0] * 10
})

protected_attribute = 'gender'
prediction_column = 'prediction'
true_label_column = 'true_label'

result = calculate_equal_opportunity(data, protected_attribute, prediction_column, true_label_column)
print(result)

# It has the same true positive rate (also called sensitivity or recall), meaning that the rate of correctly
# predicted positive instances is equal across different groups.
