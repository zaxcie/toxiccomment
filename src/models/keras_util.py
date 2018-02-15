from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback
import logging
import json
from src.data.create import create_submission


class IntervalEvaluationROCAUCScore(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            try:
                y_pred = self.model.predict(self.X_val, verbose=0)
                score = roc_auc_score(self.y_val, y_pred)
                print("Interval evaluation - epoch: {:d} - roc_auc: {:.6f}".format(epoch, score))
            except Exception as e:
                print("Error when calculating AUC")
                print(e)


def write_model_to_disk(write_path, model, configuration, y_hat_test, label_names, test_df):
    with open(write_path + "model_architecture.json", 'w') as f:
        json.dump(model.to_json(), f)

    with open(write_path + "model_config.json", "w") as f:
        json.dump(configuration, f)

    model.save_weights(write_path + "model_weight_final.h5")

    create_submission(test_df, y_hat_test, write_path, label_names)


def write_model_to_disk(write_path, model, configuration, y_hat_test, label_names, test_df, filename="submission.csv"):
    with open(write_path + "model_architecture.json", 'w') as f:
        json.dump(model.to_json(), f)

    with open(write_path + "model_config.json", "w") as f:
        json.dump(configuration, f)

    model.save_weights(write_path + "model_weight_final.h5")

    create_submission(test_df, y_hat_test, write_path, label_names, filename)
