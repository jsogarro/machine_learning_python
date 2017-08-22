from sklearn import tree


# height, weight and shoe size
X = [
    [181,80,44],
    [177,70,43],
    [160,60,38],
    [154,54,37],
    [166,65,40],
    [190,90,47],
    [175,64,39],
    [177,70,40],
    [159,55, 33],
    [171,75,42],
    [181,85,43]
]

# labels for each list in our data set
Y = ['male', 'female', 'female', 'female', 'male', 'male', 'male', 'female', 'male', 'female', 'male']


def build_decision_tree(x_values, y_values):
    '''
    Returns a trained decsion tree for a given data set

    Args:
        x_values: a list of x values in our data set
        y_values: a list of labels for data in our data set

    Returns:
        A trained classifier object
    '''
    # set our classifier to be a decision tree
    classifier = tree.DecisionTreeClassifier()

    # train the decision tree
    clf_trained = classifier.fit(x_values, y_values)
    return clf_trained


def predict_gender(observation, classifier):
    '''
    Predicts the gender of an input list and returns predicted value

    Args:
        observation: A list representing the height, weight and shoe size of a person
        classifier: The classifer to use to predict the result

    Returns:
        A string representing the label of the prediction
    '''
    prediction = classifier.predict(observation)
    return prediction


def main():
    gender_decision_tree = build_decision_tree(X, Y)
    result = predict_gender([150, 68, 39], gender_decision_tree)
    print result


if __name__ == "__main__":
    main()
