import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 생성
np.random.seed(0)
data = np.random.randn(100, 2)

# 그래픽 설정
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Exploratory Data Analysis', fontsize=16)

# 첫 번째 subplot: Mean과 Median 비교
a_mean, b_mean = np.mean(data[:, 0]), np.mean(data[:, 1])
a_median, b_median = np.median(data[:, 0]), np.median(data[:, 1])

axes[0, 0].bar(['Mean', 'Median'], [a_mean, a_median], color='blue', alpha=0.7, label='Variable 1')
axes[0, 0].bar(['Mean', 'Median'], [b_mean, b_median], color='green', alpha=0.7, label='Variable 2')
axes[0, 0].legend()
axes[0, 0].set_title('Descriptive Statistics: Mean and Median')

# 두 번째 subplot: 상관 관계 히트맵
correlation_matrix = np.corrcoef(data.T)
sns.heatmap(correlation_matrix, annot=True, ax=axes[0, 1], cmap='coolwarm')
axes[0, 1].set_title('Correlation Analysis')

# 세 번째 subplot: 히스토그램
axes[1, 0].hist(data[:, 0], bins=15, color='blue', alpha=0.7, label='Variable 1')
axes[1, 0].hist(data[:, 1], bins=15, color='green', alpha=0.7, label='Variable 2')
axes[1, 0].legend()
axes[1, 0].set_title('Histogram of Variables')

# 네 번째 subplot: 산점도
axes[1, 1].scatter(data[:, 0], data[:, 1], alpha=0.7)
axes[1, 1].set_xlabel('Variable 1')
axes[1, 1].set_ylabel('Variable 2')
axes[1, 1].set_title('Scatter Plot of Variable 1 vs Variable 2')

# 레이아웃 설정 및 그래픽 표시
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

