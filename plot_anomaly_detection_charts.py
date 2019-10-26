import pandas as pd
import matplotlib.pyplot as plt

res = pd.read_csv("anomaly_detection_results/benchmark_results.csv")
res.set_index('Model', inplace=True)  # set 'Model' columns as index

models = ['PCA', 'InvariantsMiner', 'LogClustering']
metrics = ['Precision', 'Recall', 'F1', 'Runtime']


def autolabel(ax, rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{0:.2f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    ha='center', va='bottom')


# plot bar chart of precision, recall, F1 and runtime for each model
for m in metrics:
    m_train = []
    m_test = []
    for mod in models:
        m_train.append(float(res.loc[mod + '-train', m]))
        m_test.append(float(res.loc[mod + '-test', m]))
        if m == 'Runtime':  # convert seconds to ms
            m_train[-1] = int(m_train[-1] * 1000)
            m_test[-1] = int(m_test[-1] * 1000)

    fig, ax = plt.subplots()
    rect = ax.bar(models, m_train)
    autolabel(ax, rect)
    plt.ylabel(m)
    if m == 'Runtime':
        plt.ylabel(m + ' (ms)')
    plt.xlabel('Models')
    plt.title(m + ' Comparison Training Set')
    plt.savefig("charts/" + m + "_anomaly_detection_results_train")

    fig, ax = plt.subplots()
    rect = ax.bar(models, m_test)
    autolabel(ax, rect)
    plt.ylabel(m)
    if m == 'Runtime':
        plt.ylabel(m + ' (ms)')
    plt.xlabel('Models')
    plt.title(m + ' Comparison Test Set')
    plt.savefig("charts/" + m + "_anomaly_detection_results_test")

plt.show()