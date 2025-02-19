from sklearn.model_selection import train_test_split


def splice_data(data, test_size):
    X = data.drop('increase_stock', axis=1)
    y = data['increase_stock']
    x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=test_size, random_state=42, stratify=y)
    return x_train, x_test, y_train, y_test
