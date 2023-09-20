import pandas as pd

# this function is only for binary classification

def calculate_demographic_parity(data, protected_attribute, prediction_column):
    """
    Calculate demographic parity for a binary classification problem.

    Parameters:
    data (pd.DataFrame): A dataframe containing the dataset with labels and predictions.
    protected_attribute (str): The column name of the protected attribute in the data.
    prediction_column (str): The column name of the prediction in the data.

    Returns:
    dict: A dictionary containing the demographic parity for each group.
    """

    groups = data[protected_attribute].unique()
    demographic_parity_dict = {}

    for group in groups:
        group_data = data[data[protected_attribute] == group]
        parity = group_data[prediction_column].mean()
        demographic_parity_dict[group] = parity

    return demographic_parity_dict

# multiple class
# per class

def calculate_demographic_parity_multiclass_per_class(data, protected_attribute, prediction_column):
    groups = data[protected_attribute].unique()
    classes = data[prediction_column].unique()

    demographic_parity_dict = {}

    for group in groups:
        group_data = data[data[protected_attribute] == group]
        group_parity = {}

        for c in classes:
            parity = (group_data[prediction_column] == c).mean()
            group_parity[c] = parity
        demographic_parity_dict[group] = group_parity

    return demographic_parity_dict

# overall

# consider the absolute difference in the probabilities of receiving a positive outcome
# (being classified into any of the classes) for different groups.

def calculate_overall_demographic_parity(data, protected_attribute, prediction_column):
    groups = data[protected_attribute].unique()
    demographic_parity_dict = {}

    for group in groups:
        group_data = data[data[protected_attribute] == group]
        parity = group_data[prediction_column].value_counts(normalize=True)
        demographic_parity_dict[group] = parity.to_dict()

    # Calculating overall demographic parity
    overall_parity = {}
    classes = data[prediction_column].unique()
    print(demographic_parity_dict)
    for c in classes:
        overall_parity[c] = max([demographic_parity_dict[group].get(c, 0) for group in groups]) - \
                            min([demographic_parity_dict[group].get(c, 0) for group in groups])

    return overall_parity


# Example usage:
if __name__ == "__main__":
    # Creating a simulated dataset
    data = pd.DataFrame({
        'gender': ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'F', 'F'],
        'prediction': [0, 0, 1, 1, 0, 1, 0, 0, 1, 1]
    })

    protected_attribute = 'gender'
    prediction_column = 'prediction'

    result = calculate_demographic_parity(data, protected_attribute, prediction_column)
    print('binary classification')
    print(result)

    data = pd.DataFrame({
        'gender': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'] * 10,
        'prediction': [0, 1, 2, 0, 1, 2, 0, 1, 2, 0] * 10
    })

    # Specify the column names of the protected attribute and the prediction column
    protected_attribute = 'gender'
    prediction_column = 'prediction'

    # Calculate demographic parity and print the result
    result = calculate_demographic_parity_multiclass_per_class(data, protected_attribute, prediction_column)
    print('per-class (multiple class)')
    print(result)

    data = pd.DataFrame({
        'gender': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'] * 10,
        'prediction': [0, 1, 2, 0, 1, 2, 0, 1, 2, 0] * 10
    })

    # Specify the column names of the protected attribute and the prediction column
    protected_attribute = 'gender'
    prediction_column = 'prediction'

    # Calculate demographic parity and print the result
    print('overall for multiple classes. check difference on different groups')
    result = calculate_overall_demographic_parity(data, protected_attribute, prediction_column)
    print(result)

    # Equal Demographic Parity: If the values are equal or very close to each other across groups, it indicates that the
    # classifier is achieving demographic parity; the probability of receiving a positive prediction is approximately the
    # same regardless of group membership.
    #
    # Unequal Demographic Parity: If there is a significant difference in values
    # between groups, it indicates a disparity in the predictions between the groups, meaning the classifier may be
    # biased in favor of one group over others.

