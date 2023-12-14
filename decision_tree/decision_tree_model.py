import math
import random

from decision_tree.decision_tree_extend import *



# only calculate the score on Accuracy and F1 layer
def decision_tree_l1(TP, FP, TN, FN):

    # check if the dataset is balance or not
    # check the relative difference for four terms
    # we set 0.2 for relative difference, if any comparison is larger than 0.2
    # then the dataset is not balance
    balance = check_TP_TN_FP_FN_balance(TP, TN, FP, FN, 0.2)

    # if the dataset is balance, then just run Accuracy
    # we set 0.2 as error_rate, if FP and

    cost_balance, accuracy_score, message = accuracy(TP, FP, TN, FN, 0.2)
    if balance:

        if accuracy_score >= 0.5:
            # good result, means, FP + FN <= TP + TN
            # because we care about the true prediction to true actual label
            # if you want, we can continue to check the FP and FN, but these
            # four terms are too good to analyze
            return accuracy_score
        else:
            # the low score means the FP + FN >= TP + TN, the accuracy is low
            # we definitely need to check why FP + FN >= TP + TN
            return accuracy_score #* decision_tree_l2(TP, FP, TN, FN)
    else:
        f1_score = F1(TP, FP, FN)
        # we need to check FP + FN ? TP

        if f1_score == 0:
            # too bad, it means TP = 0
            return f1_score
        elif f1_score == 1:
            # FP = FN = 0, too good, so imbalance is happened on TP nd TN
            return (f1_score + accuracy_score) / 2
        else:
            return f1_score * decision_tree_l2(TP, FP, TN, FN)


# based on the F1, we try to analyze the Precision (P) and Recall (R) ------- (FP, FN) of F1
def decision_tree_l2(TP, FP, TN, FN):

    # there are three conditions on P vs. R or FP vs. FN

    if FP > FN:

        # we need the Precision score
        precision_score = precision(TP, FP)

        # FP > FN then P < R, analyze FP in  P = TP / (TP + **FP**)
        if precision_score == 0:
            # TP = 0, too bad, no right prediction
            return 0
        elif precision_score < 0.5:
            # means TP < FP. we need to check how FP affects the other good prediction TN
            return precision_score * decision_tree_l3_FPR(FP, TN)
        elif precision_score == 0.5:
            # it means FP = TP
            # do nothing but go to l3_FP_FN
            # return precision_score * decision_tree_l3_FP_FN(FP, FN)
            return precision_score * decision_tree_l3_FP_FN_v2(FP, FN, TP, TN)
        elif precision_score == 1:
            # it is rarely possible, because FP > FN, if FP = 0, then FN is meaningless
            return 1
        else:
            # 1 > precision_score >= 0.5:
            # it means FP < TP
            # go to l3_FP_FN, it is good if TP is high, then we check FN and FP
            # return precision_score * decision_tree_l3_FP_FN(FP, FN)
            return precision_score * decision_tree_l3_FP_FN_v2(FP, FN, TP, TN)
    elif FP < FN:
        # we need the Recall score

        recall_score = recall(TP, FN)

        # FP < FN then P > R, analyze FN in R = TP / (TP + **FN**)

        if recall_score == 0:
            # TP = 0, too bad, no right prediction
            return 0
        elif recall_score < 0.5:
            # means TP < FN. we need to check how FN affects the other good prediction TN
            return recall_score * decision_tree_l3_X(FP, TN, TP)
        elif recall_score == 0.5:
            # it means TP = FN
            # do nothing but go to l3_FP_FN
            # return recall_score * decision_tree_l3_FP_FN(FP, FN)
            return recall_score * decision_tree_l3_FP_FN_v2(FP, FN, TP, TN)
        elif recall_score == 1:
            # it is rarely possible, because FP < FN, if FN = 0, then FP is meaningless
            return 1
        else:
            # 1 > recall_score >= 0.5:
            # it means FN < TP
            # go to l3_FP_FN, it is good if TP is high, then we check FN and FP
            # return recall_score * decision_tree_l3_FP_FN(FP, FN)
            return recall_score * decision_tree_l3_FP_FN_v2(FP, FN, TP, TN)

    else:

        # FP = FN
        # if F1 score = 0.5, it only means FP + FN = 2 TP
        # then we just get the average value of Recall and Precision and run
        # decision_tree_l3_FP_FN(FP, FN) to compare FN and FP
        # return recall(TP, FN) * decision_tree_l3_FP_FN(FP, FN)
        print("unchecked")
        return recall(TP, FN) * decision_tree_l3_FP_FN_v2(FP, FN, TP, TN)

def decision_tree_l3_X(FN, TN, TP):
    # X = FN / (FN + TN)
    # x1 = FN / (FN + TP)
    # we need to compare FN and TN by using X
    if FN + TN == 0:
        return 1

    x_score = X(FN, TN)
    #x_score = X1(FN, TP)
    # if TN > FN, then X < 0.5, good
    # if TN = FN, then X = 0.5, normal
    # if TN < FN, then X > 0.5, bad
    # so we have to revise the value
    return 1 - x_score

def decision_tree_l3_FPR(FP, TN):
    # FPR = FP / (FP + TN)
    # we need to compare FP and TN by using FPR

    if FP + TN == 0:
        return 1

    fpr_score = FPR(FP, TN)

    # if TN > FP, then FPR < 0.5, good
    # if TN = FP, then FPR = 0.5, normal
    # if TN < FP, then FPR > 0.5, bad
    # so we have to revise the value
    return 1 - fpr_score

def decision_tree_l3_FP_FN(FP, FN):
    # so we need to compare FP and FN

    relative_difference = abs(FP - FN) / max(FP, FN)

    # we set the relative difference <= 10 % mean they are same
    if relative_difference <= 0.1:
        return 1
    elif FN == 0 or FP == 0 or FP == FN:
        # previous setting is 1, but change to 0.5 for better balance
        return 0.5
    elif FP > FN:
        return FP / (FP + FN)
    elif FP < FN:
        return FN / (FP + FN)
    else:

        return 1


def decision_tree_l3_FP_FN_v2(FP, FN, TP, TN):
    # so we need to compare FP and FN

    # however, we need to consider that TP is the most important thing we want
    # then FP, FN are the components that we need to lower.
    # so any higher FP or FN should lower the grade
    # if FP and FN are close, then we need to consider another way to lower the grade

    # if the relative difference is low, then values are close
    relative_difference = abs(FP - FN) / max(FP, FN)

    # here, if the relative difference is high, means one value is too high, then we need to lower the grade
    # because FP and FN are negative to grade


    if FN == 0 and FP == 0:
        return 1

    elif FP == FN:
        # if FP and FN are same, then how to lower the grade, but difference is 0
        # anyway, FP and FN are negative to the grade

        # if FP is close to TP, then grade should be lowed
        # if FP is far away from TP, then grade should be higher
        # however, it doesn't compare to TP and TN, so we need to check here.
        return (TP - FP) / TP

    # if the relative difference is low, then values are close,
    # then how to deal with them
    # get the average of FP and FN,
    # then check
    # the relative difference is very important, if the value is not clearly definied, then the below return will
    # affect the grade
    elif relative_difference < 0.25:
        return (FP + FN) / (2 * (TP + TN))

    # relative difference is high, mean one of FP or FN is high
    # if FP is larger, then grade should be lower
    # if FP is lower, then grader should be higher
    else:
        return (1 - max(FP, FN) / (min(FP, FN) + max(TP, TN)))





    ''' 
    elif FP > FN:
        # if we only consider the TN, TN may be too weak, so we balance TP and TN
        return FP / (FP + FN + (TN + TP) / 2 )
    elif FP < FN:
        return FN / (FP + FN + (TN + TP) / 2 )
    '''




