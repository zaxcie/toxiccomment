from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback
import logging


class IntervalEvaluationROCAUCScore(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict_proba(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("Interval evaluation - epoch: {:d} - roc_auc: {:.6f}".format(epoch, score))
