# refers to the difference in the rates of positive predictions between different groups.

import pandas as pd


def calculate_parity_difference(data, protected_attribute, prediction_column):
    groups = data[protected_attribute].unique()

    parity_difference_dict = {}

    for i, group_a in enumerate(groups):
        for j, group_b in enumerate(groups):
            if i < j:
                prob_a = data[data[protected_attribute] == group_a][prediction_column].mean()
                prob_b = data[data[protected_attribute] == group_b][prediction_column].mean()

                parity_difference_dict[f'{group_a} vs {group_b}'] = prob_a - prob_b

    return parity_difference_dict


# Simulated dataset with predictions
data = pd.DataFrame({
    'gender': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'] * 10,
    'prediction': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 10,
})

protected_attribute = 'gender'
prediction_column = 'prediction'

result = calculate_parity_difference(data, protected_attribute, prediction_column)
print(result)

# This is the absolute difference in prediction rates between groups. A parity difference close to 0 would indicate
# that the two groups are being treated similarly by the model, while a larger parity difference indicates a
# discrepancy in treatment between the groups.

