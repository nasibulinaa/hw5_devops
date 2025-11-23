from src.load import load_and_prepare_data
from src.analyze import deepcheck_analyze, evidently_analyze
from src.train import mlflow_experiment


def main():
    df = load_and_prepare_data()
    deepcheck_analyze(df)
    evidently_analyze(df)
    mlflow_experiment(df)

if __name__ == "__main__":
    main()
