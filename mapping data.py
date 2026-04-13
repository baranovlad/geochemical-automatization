import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

data = pd.read_excel('CLS.xlsx', index_col=0)

sns.set_theme(font_scale=0.8, font='Times New Roman')
plt.figure(figsize=(6, 8), dpi=120)

plt.title('Distribution of cluster sampling points on the plane (STATISTICA)')
plot = sns.scatterplot(data, x='x msk', y='y msk', hue='true', style='true', palette='deep')
plt.savefig('map_statistica.png', dpi=300)

sns.set_theme(font_scale=0.8, font='Times New Roman')
plt.figure(figsize=(6, 8), dpi=120)
plt.title('Distribution of cluster sampling points on the plane (PYTHON)')
plot2 = sns.scatterplot(data, x='x msk', y='y msk', hue='pred', style='pred', palette='deep')
plt.savefig('map_python.png', dpi=300)
