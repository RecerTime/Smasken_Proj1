def missing_values_check(data):
    return data.isnull().values.any()