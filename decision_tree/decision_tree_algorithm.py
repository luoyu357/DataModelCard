def decision_tree_l1(TP, FP, TN, FN):
    balance = check_TP_TN_FP_FN_balance(TP, TN, FP, FN, 0.2)

    cost_balance, accuracy_score, message = accuracy(TP, FP, TN, FN, 0.2)

    if balance:
        return accuracy_score * decision_tree_l2(TP, FP, TN, FN)

    else:
        f1_score = F1(TP, FP, FN)

        if f1_score == 0:
            return f1_score
        elif f1_score == 1:
            return (f1_score + accuracy_score) / 2
        else:
            return f1_score * decision_tree_l2(TP, FP, TN, FN)


def decision_tree_l2(TP, FP, TN, FN):
    if FP > FN:
        precision_score = precision(TP, FP)

        if precision_score == 0:
            return 0
        elif precision_score < 0.5:
            return precision_score * decision_tree_l3_FPR(FP, TN)
        elif precision_score == 0.5:
            return precision_score * decision_tree_l3_FP_FN(FP, FN, TP, TN)
        elif precision_score == 1:
            return 1
        else:
            return precision_score * decision_tree_l3_FP_FN(FP, FN, TP, TN)
    elif FP < FN:
        recall_score = recall(TP, FN)

        if recall_score == 0:
            return 0
        elif recall_score < 0.5:
            return recall_score * decision_tree_l3_FNR(FN, TN)
        elif recall_score == 0.5:
            return recall_score * decision_tree_l3_FP_FN(FP, FN, TP, TN)
        elif recall_score == 1:
            return 1
        else:
            return recall_score * decision_tree_l3_FP_FN(FP, FN, TP, TN)
    else:
        return recall(TP, FN) * decision_tree_l3_FP_FN(FP, FN, TP, TN)


def decision_tree_l3_FNR(FN, TN):
    if FN + TN == 0:
        return 1

    x_score = FNR(FN, TN)

    return 1 - x_score


def decision_tree_l3_FPR(FP, TN):
    if FP + TN == 0:
        return 1
    fpr_score = FPR(FP, TN)

    return 1 - fpr_score


def decision_tree_l3_FP_FN(FP, FN, TP, TN):
    relative_difference = abs(FP - FN) / max(FP, FN)

    if FN == 0 and FP == 0:
        return 1
    elif relative_difference == 0:
        return 1 - (FP + FN) / (FP + FN + TN + TP)
    else:
        return 1 - max(FP, FN) / (max(FP, FN) + max(TP, TN))


def check_TP_TN_FP_FN_balance(TP, TN, FP, FN, unbalance_rate):
    terms = [TP, TN, FP, FN]
    for index1 in range(len(terms)):
        for index2 in range(index1 + 1, len(terms)):
            relative_difference = abs(terms[index1] - terms[index2]) / max(terms[index1], terms[index2])
            if relative_difference >= unbalance_rate:
                return False
    return True


def accuracy(TP, FP, TN, FN, error_rate):
    result = (TP + TN) / (TP + TN + FP + FN)

    if FP == FN:
        return True, result, 'FP and FN are roughly same based on the error rate ' + str(error_rate)
    else:

        relative_difference = abs(FP - FN) / max(FP, FN)

        if relative_difference <= error_rate:
            return True, result, 'FP and FN are roughly same based on the error rate ' + str(error_rate)
        else:
            return False, result, 'FP and FN are not roughly same based on the error rate ' + str(error_rate)


def precision(TP, FP):
    return TP / (TP + FP)


def recall(TP, FN):
    return TP / (TP + FN)


def F1(TP, FP, FN):
    return (2 * TP) / (2 * TP + FP + FN)


def FPR(FP, TN):
    return FP / (FP + TN)


def FNR(FN, TN):
    return FN / (FN + TN)
