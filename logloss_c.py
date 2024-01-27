import numpy as np

def calculate_logloss(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

    log_loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return log_loss

y_true = [0, 0, 1, 1]
y_pred = [0.2, 0.8, 1, 0.6]
print(calculate_logloss(y_true, y_pred))