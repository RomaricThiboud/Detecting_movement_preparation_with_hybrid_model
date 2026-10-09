# Authors: Romaric Thiboud <romaric.thiboud@inserm.fr>
#          Marine Vernet <marine.vernet@inserm.fr>
#
# License: BSD (3-clause)

import os.path as op
from pathlib import Path
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gamma
import seaborn as sns
from scipy.stats import gaussian_kde

from config import * 

path_figure = Path(path_figures, 'RT_distribution')
path_load = Path(path_analyzed_data)

if not os.path.exists(path_figure):
    os.makedirs(path_figure)

saving = False  # Whether to save the figure or not
viz_plot = True  # Whether to display the plot or not

df_name = 'RT_allsubj_project1.csv'
df_all_sub = pd.read_csv(Path(op.join(path_load, df_name)))

list_shape_human = []
list_scale_human = []
list_shape_computer = []
list_scale_computer = []
# Compute gamma distribution parameters for each participant
for subj in subjects_id:
    subj_data = df_all_sub[df_all_sub['participant'] == subj]
    rt_human = subj_data[subj_data['condition'] == 'human']['rt'].values
    rt_computer = subj_data[subj_data['condition'] == 'computer']['rt'].values
    # Fit gamma distributions
    shape_human, loc_human, scale_human = gamma.fit(rt_human, floc=0)
    shape_computer, loc_computer, scale_computer = gamma.fit(rt_computer, floc=0)
    # Append parameters to lists
    list_shape_human.append(shape_human)
    list_scale_human.append(scale_human)
    list_shape_computer.append(shape_computer)
    list_scale_computer.append(scale_computer)

# Test if the parameters of individual participants are significantly different between conditions (one-sample t-test)
diff_shape = np.array(list_shape_human) - np.array(list_shape_computer)
diff_scale = np.array(list_scale_human) - np.array(list_scale_computer)
t_shape, p_shape = stats.ttest_1samp(diff_shape, 0)
t_scale, p_scale = stats.ttest_1samp(diff_scale, 0)
print(f"Shape parameter: t={t_shape:.3f}, p={p_shape:.3f}")
print(f"Scale parameter: t={t_scale:.3f}, p={p_scale:.3f}")

average_shape_human = np.mean(list_shape_human)
average_scale_human = np.mean(list_scale_human)
average_shape_computer = np.mean(list_shape_computer)
average_scale_computer = np.mean(list_scale_computer)
std_shape_human = np.std(list_shape_human)
std_scale_human = np.std(list_scale_human)
std_shape_computer = np.std(list_shape_computer)
std_scale_computer = np.std(list_scale_computer)
print(f"Average parameters (human), shape: {average_shape_human:.3f} +- {std_shape_human:.3f}, scale: {average_scale_human:.3f} +- {std_scale_human:.3f}")
print(f"Average parameters (computer), shape: {average_shape_computer:.3f} +- {std_shape_computer:.3f}, scale: {average_scale_computer:.3f} +- {std_scale_computer:.3f}")

# 1. Set the visual style
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

# 2. Plotting the distributions 
# or fit a formal Gamma distribution.
for condition in ['human', 'computer']:
    subset = df_all_sub[df_all_sub['condition'] == condition]['rt']
    
    # Plot the histogram and KDE
    sns.histplot(subset, kde=True, element="step", stat="count", bins=40, 
                 label=f'{condition.capitalize()} Trials', alpha=0.4)

# 3. Aesthetics
plt.xlabel('Button press time (s)', fontsize=12)
plt.ylabel('Number of Trials', fontsize=12)
plt.legend()
plt.show()

################################################################################

# Compute overlap of the 2 distribution.
human = df_all_sub.loc[df_all_sub['condition'] == 'human', 'rt'].dropna()
computer = df_all_sub.loc[df_all_sub['condition'] == 'computer', 'rt'].dropna()

# KDEs
kde_h = gaussian_kde(human)
kde_c = gaussian_kde(computer)

# Common x-grid
xmin = min(human.min(), computer.min())
xmax = max(human.max(), computer.max())
x = np.linspace(xmin, xmax, 1000)

# Evaluate densities
f_h = kde_h(x)
f_c = kde_c(x)

# Overlap coefficient
overlap = np.trapezoid(np.minimum(f_h, f_c), x)

print(f"Overlap of Computer and Human trials = {overlap:.3f} ({100*overlap:.1f}%)")