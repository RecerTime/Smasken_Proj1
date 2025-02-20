from sklearn.model_selection import RandomizedSearchCV

def random_search_cv(log_reg_model,parameters, iterations):
    random_search = RandomizedSearchCV(
        log_reg_model,
        random_state=42,
        param_distributions=parameters,
        n_iter=iterations,
        n_jobs=-1,
        scoring="recall"
    )
    return random_search