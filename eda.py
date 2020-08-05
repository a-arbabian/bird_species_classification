import os
import matplotlib.pyplot as plt

DATA_DIR = '/home/ali/Documents/Datasets/225_bird_species/consolidated'
values_counts = []
for species in os.listdir(DATA_DIR):
    path = os.path.join(DATA_DIR, species)
    values_counts.append((species.capitalize(), len(os.listdir(path))))

values_counts = sorted(values_counts, key=lambda x: x[1])
labels, values = zip(*values_counts)

fig, ax = plt.subplots()
ax.bar(labels, values)
plt.xticks(rotation='vertical')
plt.show()