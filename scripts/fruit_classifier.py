from sklearn import tree


# specify our features
features = [[140, 1], [130, 1], [150, 1], [170, 0]]

# specify our labels
labels = [0, 0, 1,1]

# choose our model
clf = tree.DecisionTreeClassifier()

# train the model
clf = clf.fit(features, labels)

# predict a value
def main():
    print clf.predict([120, 1])


if __name__ == "__main__":
    main()
