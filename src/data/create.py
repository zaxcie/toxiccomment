import pandas as pd


def create_submission(df, y_hat, path, label_names, file_name = "submission.csv"):
    write_path = path + file_name

    submission_df = pd.DataFrame(columns=['id'] + label_names)
    submission_df['id'] = df['id'].values
    submission_df[label_names] = y_hat
    submission_df.to_csv(write_path, index=False)
