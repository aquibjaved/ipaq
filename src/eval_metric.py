from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from typing import List
import seaborn as sns

class EvalClass:

    def save_cf_matrix(ytrue: List, pred: List):
        """
        ytrue: ground truth
        pred: pred from model
        :return: save confusion matrix in figure
        """

        cf = confusion_matrix(ytrue, pred)
        ax = sns.heatmap(cf, annot=True, cmap='Blues')
        ax.set_title('\n\n Seaborn Confusion Matrix with labels')
        ax.set_xlabel('Predicted Values \n\n')
        ax.set_ylabel('Actual Values');
        plt.savefig("eval/cf.png")


    def roc_auc():
        pass



