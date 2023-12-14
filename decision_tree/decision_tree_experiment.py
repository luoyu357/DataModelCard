import random

from decision_tree_algorithm import *
import matplotlib.pyplot as plt
# session 1
# check TP, TN, FP, FN separately
# the max value of them is 10000
# for the checked item, each starts from 0 and is increased by 100
categories = ['TP', 'FP', 'FN', 'TN']
fixed_value = []
change_value = []

i = 0
# TP is changing

while i <= 10000:
    change_value.append(i)
    fixed_value.append(10000)
    i += 100

score_tp = []

for i in range(len(change_value)):
    score_tp.append(decision_tree_l1(change_value[i], fixed_value[i], fixed_value[i], fixed_value[i]))

# FP is changing
score_fp = []

for i in range(len(change_value)):
    score_fp.append(decision_tree_l1(fixed_value[i], change_value[i], fixed_value[i], fixed_value[i]))

# FN is changing

score_fn = []

for i in range(len(change_value)):
    score_fn.append(decision_tree_l1(fixed_value[i], fixed_value[i], fixed_value[i], change_value[i]))

# TN is changing

score_tn = []

for i in range(len(change_value)):
    score_tn.append(decision_tree_l1(fixed_value[i], fixed_value[i], change_value[i], fixed_value[i]))



fig, ax = plt.subplots()

# Set bar width
bar_width = 0.2

# Set positions for the bars
x = range(len(change_value))
x1 = [i - bar_width for i in x]
x2 = [i for i in x]
x3 = [i + bar_width for i in x]
x4 = [i + 2 * bar_width for i in x]

# Create bar plots for each set of data
plt.bar(x1, score_tp, width=bar_width, label='TP', align='center', color='deepskyblue')
plt.bar(x2, score_fp, width=bar_width, label='FP', align='center', color='black')
plt.bar(x3, score_fn, width=bar_width, label='FN', align='center', color='red')
plt.bar(x4, score_tn, width=bar_width, label='TN', align='center', color='lightseagreen')


# Set labels and title
plt.xlabel('Value of TP, FP, FN, TN')
plt.ylabel('Decision Tree score')
plt.title('Change individual value of TP, FP, FN, TN while others are fixed value of 10000')

location = []
value = []
lo = 0
while lo < 101:
    location.append(lo)
    value.append(change_value[lo])
    lo += 10

# Set x-axis tick labels
#plt.xticks(x, change_value)
plt.xticks(location, value)
# Display a legend
plt.legend()

plt.savefig('individual_check.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.tight_layout()
plt.show()


# session 2
# randomly generate 5000 values for them, range 0 - 10000


TP = [random.randint(0, 10000) for i in range(5000)]
FP = [random.randint(0, 10000) for i in range(5000)]
TN = [random.randint(0, 10000) for i in range(5000)]
FN = [random.randint(0, 10000) for i in range(5000)]

# Mock performance as a function of params
performance = []
for i in range(0, 5000):
    performance.append(decision_tree_l1(TP[i], FP[i], TN[i], FN[i]))


# Plotting
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot a 3D scatter plot
ax.scatter(TP, FP, performance, c=performance, cmap='viridis', marker='o')

ax.set_xlabel('TP')
ax.set_ylabel('FP')
ax.set_zlabel('Decision Tree Score')

plt.title('The correlation between TP and FP in the Decision Score')
plt.savefig('TP_FP_3D.png', dpi=300, bbox_inches='tight')
plt.show()



# Plotting
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot a 3D scatter plot
ax.scatter(TP, FN, performance, c=performance, cmap='viridis', marker='o')

ax.set_xlabel('TP')
ax.set_ylabel('FN')
ax.set_zlabel('Decision Tree Score')

plt.title('The correlation between TP and FN in the Decision Score')
plt.savefig('TP_FN_3D.png', dpi=300, bbox_inches='tight')
plt.show()
# Plotting
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')


# Plot a 3D scatter plot
ax.scatter(TP, TN, performance, c=performance, cmap='viridis', marker='o')

ax.set_xlabel('TP')
ax.set_ylabel('TN')
ax.set_zlabel('Decision Tree Score')

plt.title('The correlation between TP and TN in the Decision Score')
plt.savefig('TP_TN_3D.png', dpi=300, bbox_inches='tight')
plt.show()


# Plotting
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot a 3D scatter plot
ax.scatter(FP, FN, performance, c=performance, cmap='viridis', marker='o')

ax.set_xlabel('FP')
ax.set_ylabel('FN')
ax.set_zlabel('Decision Tree Score')

plt.title('The correlation between FP and FN in the Decision Score')
plt.savefig('FP_FN_3D.png', dpi=300, bbox_inches='tight')
plt.show()





# Plotting
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot a 3D scatter plot
ax.scatter(FP, TN, performance, c=performance, cmap='viridis', marker='o')

ax.set_xlabel('FP')
ax.set_ylabel('TN')
ax.set_zlabel('Decision Tree Score')

plt.title('The correlation between FP and TN in the Decision Score')
plt.savefig('FP_TN_3D.png', dpi=300, bbox_inches='tight')
plt.show()


# Plotting
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot a 3D scatter plot
ax.scatter(FN, TN, performance, c=performance, cmap='viridis', marker='o')

ax.set_xlabel('FN')
ax.set_ylabel('TN')
ax.set_zlabel('Decision Tree Score')

plt.title('The correlation between FN and TN in the Decision Score')
plt.savefig('FN_TN_3D.png', dpi=300, bbox_inches='tight')
plt.show()


