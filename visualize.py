from sklearn.metrics import confusion_matrix
import numpy as np

y_true, y_pred = np.load('results.npy', allow_pickle=True)
labels = ['STANDING', 'SITTING', 'LAYING', 'WALKING', 'WALKING_DOWNSTAIRS', 'WALKING_UPSTAIRS']

confusion = confusion_matrix(y_true, y_pred)
from sklearn.metrics import classification_report
print('\nClassification Report\n')
print(classification_report(y_true, y_pred, target_names=labels))