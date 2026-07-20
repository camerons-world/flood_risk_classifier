TRAINING_START = "2010-01-01"


def assemble_training_set():
    raise NotImplementedError("pulls Hydrology API readings + Historic Flood Warnings labels")


def train_baseline_model(X, y):
    raise NotImplementedError("class-weighted logistic regression")


def train_main_model(X, y):
    raise NotImplementedError("class-weighted XGBoost")


if __name__ == "__main__":
    assemble_training_set()
