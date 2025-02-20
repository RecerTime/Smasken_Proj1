from sklearn.linear_model import LogisticRegression

def create_logistic_model(x_train, y_train, max_iter, random_state):
    log_reg_model = LogisticRegression(max_iter=max_iter, random_state=random_state)
    log_reg_model.fit(x_train, y_train)
    return log_reg_model