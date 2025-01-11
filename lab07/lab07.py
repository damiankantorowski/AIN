import csv
import matplotlib.pyplot as plt
import numpy as np

def naive_pareto_front(evaluations):
    pareto_fronts = []
    new_evaluations = evaluations.copy()

    while len(evaluations) != 0:
        pareto_front = []
        for i in range(len(evaluations)):
            dominated = False
            for j in range(len(evaluations)):
                if (
                    i == j or
                    evaluations[i][0] < evaluations[j][0] or
                    evaluations[i][1] > evaluations[j][1] or
                    (evaluations[i][0] == evaluations[j][0] and evaluations[i][1] == evaluations[j][1])
                ):
                    continue
                else:
                    dominated = True
                    break
            if not dominated:
                pareto_front.append(evaluations[i])
                new_evaluations.remove(evaluations[i])

        pareto_fronts.append(pareto_front)
        evaluations = new_evaluations.copy()

    return pareto_fronts

def kung_pareto_front(evaluations):
    def Front(P):
        if len(P) == 1:
            return P
        else:
            middle = len(P) // 2
            T = Front(P[:middle])
            B = Front(P[middle:])
            P_new = T
            for point_B in B:
                is_dominated = False
                for point_T in T:
                    if point_T[1] > point_B[1]:
                        is_dominated = True
                        break
                if not is_dominated:
                    P_new.append(point_B)
            return P_new

    evaluations = sorted(evaluations)
    pareto_fronts = []
    while len(evaluations) != 0:
        pareto_fronts.append(Front(evaluations))
        evaluations = [point for point in evaluations if point not in pareto_fronts[-1]]

    return pareto_fronts

evaluations = []
with open('MO-D3R.txt', 'r') as f:
    for line in f:
        f1, f2 = map(float, line.strip().split('\t'))
        evaluations.append((f1, f2))

# Compute Pareto fronts
naive_fronts = naive_pareto_front(evaluations.copy())
kung_fronts = kung_pareto_front(evaluations.copy())

naive_fronts = [sorted(front) for front in naive_fronts] # Sort points in each front according to f1, to make comparison easier in txt files between Kung and naive fronts

# Write Pareto fronts to files
write_file = open('pareto_fronts_Kung.txt', 'w')
write_file.write('nr of fronts: ' + str(len(kung_fronts)))
write_file.write('\n')
for front in kung_fronts:
    for point in front:
        write_file.write(str(point[0]) + '\t' + str(point[1]) + '\n')
    write_file.write('\n\n')

write_file.close()

write_file = open('pareto_fronts_naive.txt', 'w')
write_file.write('nr of fronts: ' + str(len(naive_fronts)))
write_file.write('\n')
for front in naive_fronts:
    for point in front:
        write_file.write(str(point[0]) + '\t' + str(point[1]) + '\n')
    write_file.write('\n\n')

write_file.close()

print("nr of Kung fronds:", len(kung_fronts), "\n nr of naive fronts:", len(naive_fronts))

# Plot Pareto fronts
n = 5  # Number of fronts to highlight
colors = plt.cm.rainbow(np.linspace(0, 1, n))

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

fig.suptitle('Minimize f1, Maximize f2', fontsize=16)

# Plot Naive Pareto fronts
for i, (pareto_front, color) in enumerate(zip(naive_fronts[:n], colors)):
    pareto_x = [point[0] for point in pareto_front]
    pareto_y = [point[1] for point in pareto_front]
    axs[0].scatter(pareto_x, pareto_y, color=color, label=f'Front {i+1}')

for pareto_front in naive_fronts[n:]:
    pareto_x = [point[0] for point in pareto_front]
    pareto_y = [point[1] for point in pareto_front]
    axs[0].scatter(pareto_x, pareto_y, c='black', marker='.')

axs[0].set_title('Naive Pareto Fronts')
axs[0].set_xlabel('f1')
axs[0].set_ylabel('f2')
axs[0].legend()

# Plot Kung Pareto fronts
for i, (pareto_front, color) in enumerate(zip(kung_fronts[:n], colors)):
    pareto_x = [point[0] for point in pareto_front]
    pareto_y = [point[1] for point in pareto_front]
    axs[1].scatter(pareto_x, pareto_y, color=color, label=f'Front {i+1}')

for pareto_front in kung_fronts[n:]:
    pareto_x = [point[0] for point in pareto_front]
    pareto_y = [point[1] for point in pareto_front]
    axs[1].scatter(pareto_x, pareto_y, c='black', marker='.')

axs[1].set_title('Kung Pareto Fronts')
axs[1].set_xlabel('f1')
axs[1].set_ylabel('f2')
axs[1].legend()

plt.tight_layout()
plt.show()
