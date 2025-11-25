from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def train_decision_tree(X_train, X_test, y_train, y_test) -> None:
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    accuracy = accuracy_score(pred, y_test)
    print('Accuracy :\033[32m \033[01m {:.2f}% \033[30m \033[0m'.format(accuracy*100))
    print('Classification_report')
    print(classification_report(y_test, pred))
    print('Confusion_matrix:')
    cf_matrix = confusion_matrix(y_test, pred)
    plot_ = sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True,fmt= '0.2%')
    plt.show()