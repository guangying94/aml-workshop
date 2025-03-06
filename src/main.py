# imports
import mlflow
import argparse
import mltable
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# define functions
def main(args):
    """
    Main function to set up MLflow, process data, and train the model.
    """
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.autolog()

    tbl = mltable.load(args.input)
    df = tbl.to_pandas_dataframe()

    X_train, X_test, y_train, y_test = process_data(df, args.test_size, args.random_state)
    model = train_model(X_train, X_test, y_train, y_test, args.random_state)


def process_data(df, test_size, random_state):
    """
    Process the input dataframe by converting categorical columns, splitting the data,
    and standardizing the features.
    """
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
    X = df.drop(columns=['Survived', 'Name', 'Ticket', 'Cabin'])
    y = df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def train_model(X_train, X_test, y_train, y_test, random_state):
    """
    Train a logistic regression classifier.
    """
    model = LogisticRegression(random_state=random_state)
    model = model.fit(X_train, y_train)

    return model


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help='mltable to read')
    parser.add_argument("--mlflow_uri", type=str)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.15)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)