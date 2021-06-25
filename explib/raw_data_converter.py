import numpy as np
import pandas as pd
import pickle as pkl


# preprocess for SWaT. SWaT.A2_Dec2015, version 0
df = pd.read_csv('SWaT_Dataset_Attack_v0.csv')
y = df['Normal/Attack'].to_numpy()
labels = []
for i in y:
    if i == 'Attack':
        labels.append(1)
    else:
        labels.append(0)
labels = np.array(labels)
assert len(labels) == 449919
# pkl.dump(labels, open('SWaT_test_label.pkl', 'wb'))
print('SWaT_test_label saved')

df = df.drop(columns=[' Timestamp', 'Normal/Attack'])
test = df.to_numpy()
assert test.shape == (449919, 51)
# pkl.dump(test, open('SWaT_test.pkl', 'wb'))
print('SWaT_test saved')

df = pd.read_csv('SWaT_Dataset_Normal_v0.csv')
df = df.drop(columns=['Unnamed: 0','Unnamed: 52'])
train = df[1:].to_numpy()
assert train.shape == (496800, 51)
# pkl.dump(train, open('SWaT_train.pkl', 'wb'))
print('SWaT_train saved')

# preprocess for WADI. WADI.A1
a = str(open('WADI_14days.csv', 'rb').read(), encoding='utf8').split('\n')[5: -1]
a = '\n'.join(a)
with open('train1.csv', 'wb') as f:
    f.write(a.encode('utf8'))
a = pd.read_csv('train1.csv', header=None)


a = a.to_numpy()[:, 3:]
nan_cols = []
for j in range(a.shape[1]):
    for i in range(a.shape[0]):
        if a[i][j] != a[i][j]:
            nan_cols.append(j)
            break
# len(nan_cols) == 9
train = np.delete(a, nan_cols, axis=1)
assert train.shape == (1209601, 118)
# pkl.dump(train, open('WADI_train.pkl', 'wb'))
print('WADI_train saved')

df = pd.read_csv('WADI_attackdata.csv')
test = df.to_numpy()[:, 3:]
test = np.delete(test, nan_cols, axis=1)
assert test.shape == (172801, 118)
# pkl.dump(test, open('WADI_test.pkl', 'wb'))
print('WADI_test saved')

print('WADI_test_label saved')

# WADI labels.pkl are created manually via the description file of the dataset
