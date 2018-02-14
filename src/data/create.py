import pandas as pd
import os
from modeldb.basic.ModelDbSyncerBase import *
from datetime import datetime


def create_submission(df, y_hat, path, label_names, file_name = "submission.csv"):
    write_path = path + file_name

    submission_df = pd.DataFrame(columns=['id'] + label_names)
    submission_df['id'] = df['id'].values
    submission_df[label_names] = y_hat
    submission_df.to_csv(write_path, index=False)


def create_model(config):
    model_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    folder_model_path = config["ModelPath"] + "/" + model_name
    os.mkdir(folder_model_path)

    return model_name


def add_modeldb_entry(config, saved_model_info):
    syncer_obj = Syncer.create_syncer(config["ModelDBProjectName"],
                                      config["ModelDBProjectAuthor"],
                                      config["ModelDBProjectDescription"])

    datasets = {
        "train": Dataset(config["TrainDataPath"]),
        "val": Dataset(config["ValDataPath"]),
        "test": Dataset(config["TestDataPath"])
    }

    # create the Model, ModelConfig, and ModelMetrics instances
    model = saved_model_info["ModelName"]
    model_type = config["ModelType"]
    mdb_model1 = Model(model_type, model, "/path/to/model1")
    model_config1 = ModelConfig(model_type)
    model_metrics1 = ModelMetrics({"accuracy": saved_model_info["accuracy"],
                                   "loss": saved_model_info["loss"],
                                   "AUC": saved_model_info["AUC"]})

    # sync the datasets to modeldb
    syncer_obj.sync_datasets(datasets)

    # sync the model with its model config and specify which dataset tag to use for it
    syncer_obj.sync_model("train", model_config1, mdb_model1)

    # sync the metrics to the model and also specify which dataset tag to use for it
    syncer_obj.sync_metrics("test", mdb_model1, model_metrics1)

    syncer_obj.sync()