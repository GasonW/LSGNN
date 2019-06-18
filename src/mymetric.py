from sklearn.metrics import precision_score, recall_score, f1_score, log_loss, accuracy_score
from sklearn.metrics import confusion_matrix


def summary(y_true, y_pred):
    precision = precision_score(y_true,y_pred)
    recall = recall_score(y_true,y_pred)
    f1 = 2*precision*recall/(precision+recall)
    acc = accuracy_score(y_true, y_pred)

    print("accuracy: %.4f" %(acc))
    print("precision: %.4f" % (precision))
    print("recall: %.4f" % (recall))
    print("f1_score: %.4f" %f1)

def my_confusion_matrix(y_true, y_pred):
    labels = list(set(y_true))
    conf_mat = confusion_matrix(y_true,y_pred,labels=labels)

    print("=== Confusion Matrix ===")
    print("%-6s" %'0',"%-6s" %'1',"  <----predict as")
    print("%-6s"%conf_mat[0][0],"%-6s"%conf_mat[0][1]," | 0")
    print("%-6s"%conf_mat[1][0],"%-6s"%conf_mat[1][1]," | 1")