# need to calculate

# Prediction    Actual
# True          Positive = TP
# False         Positive = FP       incorrect prediction to positive class
# False         Negative = FN       incorrect prediction to negative class
# True          Negative = TN

# goal:
#TP and TN:
# Both these values contribute positively to the accuracy as they represent the correct predictions made by the model.
# Maximizing these values will improve accuracy.

#FP and FN: These values represent incorrect predictions made by the model.
# Minimizing these values will improve accuracy.

# Accuracy = ( TP + TN) / ( TP + TN + FP + FN)
# Precision = TP / ( TP + FP )
# Recall = TP / ( TP + FN )
# F1 = ( 2 * Precision * Recall ) / ( Precision + Recall ) = (2 * TP) / (2 * TP + FP + FN)

# advanced idea
# True Positive Rate = TP / ( TP + FN )
# True Negative Rate = TN / ( TN + FP )
# False Positive Rate = FP / ( FP + TN )
# False Negative Rate = FN / ( FN + TP)

# ? Log loss: It measures the dissimilarity between predicted probabilities and actual class labels. The objective is
# to minimize this loss function, thereby improving the accuracy and reliability of the classification model.

from sklearn.metrics import confusion_matrix


def perf_measure(y_actual, y_pred):
    cm = confusion_matrix(y_actual, y_pred)

    # Step 3: Extract TP, FP, FN, TN
    TN, FP, FN, TP = cm.ravel()

    # Step 4: Print the values
    print(f"True Positives (TP): {TP}")
    print(f"False Positives (FP): {FP}")
    print(f"False Negatives (FN): {FN}")
    print(f"True Negatives (TN): {TN}")

    return TP, FP, TN, FN

y_actual = [1, 0, 1, 1, 0, 1, 0, 0, 0, 1]
y_pred = [1, 0, 1, 0, 0, 1, 1, 0, 0, 0]


# 1. decide which one has the highest privilege
# 2. decide the logic to build the tree for these 4 functions

# 1. Accuracy
# Importance: Generally useful when the classes are balanced and the costs of false positives and false negatives are roughly the same.
# Limitation: Can be misleading if the dataset is imbalanced.

# check the input data: if classes are balances
# check the output (predication/actual): if classes are balances
# data should be saved in the dataframe format

import pandas as pd

# create a dataframe
# with 5 rows and 4 columns
data = pd.DataFrame({
    'name': ['sravan', 'ojsawi', 'bobby', 'rohith',
             'gnanesh', 'sravan', 'sravan', 'ojaswi'],
    'age': [22, 22, 23, 23, 23, 21, 21, 21]
})


def check_class_balance(data, column_name, unbalance_rate):
    data_counter = data[column_name].value_counts()
    # mathematically check the count of each class if ==
    if data_counter.nunique() == 1:
        return True
    else:
        # set the average value, and check the unbalance rate for each class
        average = sum([i[1] for i in data_counter.items()]) / len(data_counter)
        for item in data_counter.items():
            if average * (1 - unbalance_rate) <= item[1] <= average * (1 + unbalance_rate):
                continue
            else:
                return False
        return True




def check_TP_TN_FP_FN_balance(TP, TN, FP, FN, unbalance_rate):
    terms = [TP, TN, FP, FN]
    for index1 in range(len(terms)):
        for index2 in range(index1+1, len(terms)):
            relative_difference = abs(terms[index1] - terms[index2]) / max(terms[index1], terms[index2])
            if relative_difference >= unbalance_rate:
                return False
    return True


def accuracy(TP, FP, TN, FN, error_rate):
    result = (TP + TN) / (TP + TN + FP + FN)

    # check FP and FN if they are roughly the same
    # we cannot determine the cost of FP and FN, but we can use math function to see
    # if their values are same or not by adding the error_rate
    if FP == FN:
        return True, result, 'FP and FN are roughly same based on the error rate ' + str(error_rate)
    else:

        relative_difference = abs(FP - FN) / max(FP, FN)

        if relative_difference <= error_rate:
            return True, result, 'FP and FN are roughly same based on the error rate ' + str(error_rate)
        else:
            return False, result, 'FP and FN are not roughly same based on the error rate ' + str(error_rate)




# 2. Precision
# Importance: More important in situations where minimizing false positives is the primary concern. For instance,
# in email spam detection, you'd want to minimize the chance of marking a legitimate email as spam (high precision).
# Limitation: It does not consider false negatives.

# if FP is the primary concern
# ***** it doesn't consider the FN, so we need to check the F1 before using the Precision *****

def precision(TP, FP):
    return TP / ( TP + FP )

# 3. Recall
# Importance: More important in situations where minimizing false negatives is the primary concern.
# For instance, in medical diagnostics, you'd want to catch as many true positive cases as possible,
# even if it means having more false positives (high recall).
# Limitation: It does not consider false positives.

# if FN is the primary concern
# ***** it doesn't consider the FP, so we need to check the F1 before using the Precision *****

def recall(TP, FN):
    return TP / ( TP + FN )

# 4. F1-Score
# Importance: Useful when you want to balance precision and recall, especially in the case of an imbalanced dataset.
# It gives a single metric that balances both concerns.
# Limitation: It is a composite metric, so sometimes optimizing for F1-score can mask individual issues with precision or recall.

# so we run the F1 before run the recall and precision
# if F1 is low, then FP and FN is higher, mean the incorrect prediction is higher


def F1(TP, FP, FN):
    return (2 * TP) / (2 * TP + FP + FN)



def FPR(FP, TN):
    return FP / (FP + TN)

def X(FN, TN):
    return FN / (FN + TN)

def X1(FN, TP):
    return FN / (FN + TP)


# 1. build the decision tree architecture
# 2. build a dynamic tree weight for the tree travel

# step 1:
# so for Precision, F1 and Recall, we need to check the FP and FN. if they are lower, mean incorrect prediction is low




# we don't know how low of FP and FN, so run F1 to see the situation



# F1: 0 - 0.5: poor
# check FP and FN, run Precision for FP and run Recall for FN.

# 0.5 - 0.7: fair to good
# 0.7 - 0.9: good to very good
# 0.9 - 1: perfect