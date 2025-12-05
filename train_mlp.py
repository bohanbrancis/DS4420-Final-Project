import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from data_clean import load_and_prepare_nfl_data


def main():
    # create train and split
    df, feature_cols = load_and_prepare_nfl_data()

    # Temporal split by season
    train = df[df["season"] <= 2020]
    val   = df[df["season"] == 2021]
    test  = df[df["season"] >= 2022]

    X_train = train[feature_cols].values.astype(float)
    y_train = train["y"].values.astype(int)

    X_val = val[feature_cols].values.astype(float)
    y_val = val["y"].values.astype(int)

    X_test = test[feature_cols].values.astype(float)
    y_test = test["y"].values.astype(int)


    # Standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)


    # create model
    input_dim = X_train_scaled.shape[1]

    model = Sequential([
        Dense(64, activation="relu", input_shape=(input_dim,)),
        Dropout(0.4),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy"
    )

    model.summary()

    # Class imbalance handling
    pos = y_train.sum()
    neg = len(y_train) - pos
    pos_weight = neg / pos if pos > 0 else 1.0
    class_weight = {0: 1.0, 1: float(pos_weight)}
    print("Positive class weight:", pos_weight)

    # Early stopping
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    )

    # train model
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=40,
        batch_size=256,
        class_weight=class_weight,
        callbacks=[early_stop],
        verbose=1
    )

    # run model
    val_pred = model.predict(X_val_scaled).ravel()
    test_pred = model.predict(X_test_scaled).ravel()

    val_auprc = average_precision_score(y_val, val_pred)
    val_auc = roc_auc_score(y_val, val_pred)
    test_auprc = average_precision_score(y_test, test_pred)
    test_auc = roc_auc_score(y_test, test_pred)

    print(f"Validation AUPRC: {val_auprc:.4f}")
    print(f"Validation AUROC: {val_auc:.4f}")
    print(f"Test AUPRC: {test_auprc:.4f}")
    print(f"Test AUROC: {test_auc:.4f}")


if __name__ == "__main__":
    main()