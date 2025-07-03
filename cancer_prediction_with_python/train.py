from sklearn.model_selection import GridSearchCV
from sklearn import metrics

def tune_and_train_model(clf, X_train, y_train, X_test, y_test):
    param_grid = {
        'criterion': ['gini', 'entropy', 'log_loss'],
        'max_depth': [None, 3, 5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [None, 'sqrt', 'log2']
    }

    grid_search = GridSearchCV(estimator=clf,
                               param_grid=param_grid,
                               cv=5,
                               scoring='accuracy',
                               n_jobs=-1,
                               verbose=1)

    grid_search.fit(X_train, y_train)

    print("Best parameters found: ", grid_search.best_params_)
    print("Best cross-validation accuracy: {:.2f}".format(grid_search.best_score_))

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    print("Test set accuracy: {:.2f}".format(metrics.accuracy_score(y_test, y_pred)))
    return best_model
