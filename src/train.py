# /usr/bin/env python3
import joblib
from pathlib import Path
import numpy as np

from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler
from sklearn.ensemble import GradientBoostingRegressor

from azureml.core import Run, Model
from azureml.exceptions import WebserviceException


def train(run, train_df, test_df):
    train_labels = train_df.pop("departures")
    test_labels = test_df.pop("departures")

    regressor = GradientBoostingRegressor(verbose=True)
    ct = make_column_transformer(
        (MaxAbsScaler(), make_column_selector(dtype_include=np.number)),
        (OneHotEncoder(), make_column_selector(dtype_include=object)),
    )

    pipeline = Pipeline([("ColumnTransformer", ct), ("Regressor", regressor)])
    pipeline.fit(train_df, train_labels)
    y_ = pipeline.predict(test_df)

    rmse = mean_squared_error(test_labels, y_, squared=False)
    r2 = r2_score(test_labels, y_)
    print("rmse", rmse)
    print("r2", r2)

    run.log("rmse", rmse)
    run.log("r2", r2)
    return pipeline, r2


def main():
    run = Run.get_context()

    datasets = run.input_datasets
    train_dataset = datasets["train_ds"]
    test_dataset = datasets["test_ds"]

    model, r2 = train(
        run.parent,
        train_dataset.to_pandas_dataframe(),
        test_dataset.to_pandas_dataframe(),
    )

    filename = "departure-regressor.pkl"
    path = str(Path("./outputs", filename))
    model_name = "demo-departure-regressor"
    joblib.dump(model, filename=path)

    ws = run.experiment.workspace
    try:
        old_model = Model(ws, name=model_name)
        old_metrics = float(old_model.tags.get("r2"))
    except WebserviceException as e:
        print(e)
        old_metrics = np.inf

    if old_metrics >= r2:
        run.upload_file(path, path)
        run.register_model(
            model_name=model_name,
            model_path=path,
            description=" Regression Model for station depertures",
            model_framework="ScikitLearn",
            datasets=[
                ("departure test dataset", train_dataset),
                ("departure train dataset", test_dataset),
            ],
            tags={**run.tags, "r2": r2},
        )


if __name__ == "__main__":
    main()
