## Demographic Parity
* What it compares: Predictions between different groups (true values are ignored)
* Reason to use: If the input data are known to contain biases, demographic parity may be appropriate to measure fairness
* Caveats: By only using the predicted values, information is thrown away. The selection rate is also a very coarse measure of the distribution between groups, making it tricky to use as an optimization constraint

## Equalized Odds
* What it compares: True and False Positive rates between different groups
* Reason to use: If historical data does not contain measurement bias or historical bias that we need to take into account, and true and false positives are considered to be (roughly) of the same importance, equalized odds may be useful
* Caveats: If there are historical biases in the data, then the original labels may hold little value. A large imbalance between the positive and negative classes will also accentuate any statistical issues related to sensitive groups with low membership

## Equal opportunity
* What it compares: True Positive rates between different groups
* Reason to use: If historical data are useful, and extra false positives are much less likely to cause harm than missed true positives, equal opportunity may be useful
* Caveats: If there are historical biases in the data, then the original labels may hold little value. A large imbalance between the positive and negative classes will also accentuate any statistical issues related to sensitive groups with low membership

## Parity difference 
* What it compares: comparing the parity (even or odd) of two numbers
* Reason to use: Determine if two numbers have the same or different parities
* Caveats: only have two values: 0 (if both numbers have the same parity) or 1 (if the numbers have different parities)