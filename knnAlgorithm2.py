import csv
import datetime
import math
import operator
import random

import pandas_datareader.data as dr
import pandas
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split


def main():
    startDate = datetime.datetime(2014, 1, 1)
    endDate = datetime.date.today()
    # predict stock tendance of Amazon
    predict(10, 'AmazonHistoricalQuotes.csv', "AMZN", startDate, endDate)


# k = so hang xom
def predict(k, fileName, companyStockName, startDate, endDate):
    attributes = ["date", "open", "high", "low", "close", "state"]
    trainingSet = []
    testSet = []
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    totalSet = 0
    getStockDataAndWriteToFile(fileName, companyStockName, startDate, endDate)

    loadData(fileName, trainingSet, testSet, X_train, y_train, X_test, y_test, attributes)

    print("Predicting for", companyStockName)
    print("Training set: " + repr(len(y_train)))
    print("Test set: " + repr(len(y_test)))
    totalSet += len(trainingSet) + len(testSet)
    print("Total: " + repr(totalSet))

    # X_train = X_train[0]
    # y_train = y_train[0]
    # X_test = X_test[0]
    # y_test = y_test[0]

    noUpInTrainSets = 0
    noDownInTrainSets = 0
    for i in range(len(y_train)):
        if y_train[i] == "up":
            noUpInTrainSets += 1
        else:
            noDownInTrainSets += 1

    p_of_noUpInTrainSets = noUpInTrainSets / len(y_train)
    p_of_noDownInTrainSets = noDownInTrainSets / len(y_train)
    predictAndGetAccuracy(trainingSet, testSet, X_train, y_train, X_test, y_test, k, companyStockName,
                          p_of_noUpInTrainSets,
                          p_of_noDownInTrainSets)


def getStockDataAndWriteToFile(fileName, companyStockName, startDate, endDate):
    stockData = dr.DataReader(companyStockName, 'yahoo', startDate, endDate)
    stockJson = stockData.to_json(orient="index", date_format='iso')
    stockDates = json.loads(stockJson)

    firstLineOfData = True
    with open(fileName, 'w', newline='') as stockFile:
        stockWriter = csv.writer(stockFile)
        sortedDate = sorted(stockDates.keys())
        for i in sortedDate:
            formatDate = i[:10]
            if firstLineOfData:
                prevClose = stockDates[i]["Adj Close"]
                firstLineOfData = False
                continue
            stockWriter.writerow(
                [formatDate] + [stockDates[i]["Open"]] + [stockDates[i]["High"]] + [stockDates[i]["Low"]] + [
                    stockDates[i]["Adj Close"]] + [change(stockDates[i]["Adj Close"], prevClose)])
            prevClose = stockDates[i]['Adj Close']


def change(today, yesterday):
    if today > yesterday:
        return 'up'
    return 'down'


def loadData(fileName, trainingSet=[], testSet=[], X_train=[], y_train=[], X_test=[], y_test=[], attributes=[]):
    with open(fileName, 'r') as stockFile:
        stock_X = []
        stock_y = []
        lines = csv.reader(stockFile)
        split = 0.67
        dataset = list(lines)
        for x in range(len(dataset)):
            for y in range(1, len(attributes) - 1):
                dataset[x][y] = float(dataset[x][y])
            stock_y.append(dataset[x][-1])
            stock_X.append(dataset[x][: len(dataset[x]) - 1])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])

        for y in range(len(trainingSet)):
            X_train.append(trainingSet[y][: len(trainingSet[y]) - 1])
            y_train.append(trainingSet[y][-1])

        for z in range(len(testSet)):
            X_test.append(testSet[z][: len(testSet[z]) - 1])
            y_test.append(testSet[z][-1])

        # Xtrain, Xtest, ytrain, ytest = train_test_split(stock_X, stock_y, test_size=0.33)
        # X_train.append(Xtrain)
        # y_train.append(ytrain)
        # X_test.append(Xtest)
        # y_test.append(ytest)


def predictAndGetAccuracy(trainingSet, testSet, X_train, y_train, X_test, y_test, k, companyStockName,
                          p_of_noUpInTrainSets,
                          p_of_noDownInTrainSets):
    predictions = []
    predictions_knn_probabilistic = []
    for x in range(len(y_test)):
        neighbors = getNeighbors(X_train, y_train, X_test[x], y_test[x], k)
        result = getResponse(neighbors)
        result_knn_probabilistic = getResponse_knn_probabilistic(neighbors, p_of_noUpInTrainSets,
                                                                 p_of_noDownInTrainSets)
        predictions.append(result)
        predictions_knn_probabilistic.append(result_knn_probabilistic)

    accuracy = getAccuracy(y_test, predictions)
    accuracy_of_knn_prob = getAccuracy(y_test, predictions_knn_probabilistic)
    # print('Similarty: ' + repr(getAccuracy(predictions, predictions_knn_probabilistic)) + '%')
    # print('Accuracy: ' + repr(accuracy) + '%')
    print('Accuracy of knn probabilistic: ' + repr(accuracy_of_knn_prob) + '%')

    # drawing another
    plt.figure(1)
    plt.title("Prediction trend of Amazon")
    x = []
    y = []
    p = 0
    for dates in range(len(testSet)):
        if dates < 10:
            p += 1
            new_date = datetime.datetime.strptime(testSet[dates][0], "%Y-%M-%d")
            # row.append(new_date)
            x.append(p)
            if predictions[dates] == "down":
                y.append(-1)
            else:
                y.append(1)
    plt.plot(x, y, 'r', label="Predicted Trend")
    plt.show()

    plt.figure(2)
    plt.title("Actual trend of Amazon")
    x = []
    y = []
    for dates in range(len(testSet)):
        if dates < 10:
            new_date = datetime.datetime.strptime(testSet[dates][0], "%Y-%M-%d")
            x.append(new_date)
            if testSet[dates][-1] == "down":
                y.append(-1)
            else:
                y.append(1)
    plt.plot(x, y, 'b', label="Actual Trend")
    plt.show()

    plt.figure(3)
    plt.title("Prediction vs Actual Trend of Amazon ")
    plt.legend(loc="best")
    row = []
    col = []
    for dates in range(len(testSet)):
        new_date = datetime.datetime.strptime(testSet[dates][0], "%Y-%M-%d")
        row.append(new_date)
        if predictions[dates] == "down":
            col.append(-1)
        else:
            col.append(1)
    predicted_plt, = plt.plot(row, col, 'r', label="Predicted Trend")

    row = []
    col = []
    for dates in range(len(testSet)):
        new_date = datetime.datetime.strptime(testSet[dates][0], "%Y-%M-%d")
        row.append(new_date)
        if testSet[dates][-1] == "down":
            col.append(-1)
        else:
            col.append(1)
    actual_plt, = plt.plot(row, col, 'b', label="Actual Trend")
    plt.legend(handles=[predicted_plt, actual_plt])
    plt.show()


def getNeighbors(X_train, y_train, X_test, y_test, k):
    distance = []
    for x in range(len(y_train)):
        dist = euclideanDistance(X_train[x], y_train[x], X_test, y_test)
        distance.append((X_train[x], y_train[x], dist))
    distance.sort(key=operator.itemgetter(2))
    neighbors = []
    for x in range(k):
        neighbors.append((distance[x][0], distance[x][1]))
    return neighbors


def euclideanDistance(X_train, y_train, X_test, y_test):
    distance = 0
    for x in range(1, len(X_train)):
        distance += pow((X_train[x] - X_test[x]), 2)

    return math.sqrt(distance)


def getResponse(neighbors):
    votes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][1]
        if response in votes:
            votes[response] += 1
        else:
            votes[response] = 1
    sortedVotes = sorted(votes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def getResponse_knn_probabilistic(neighbors, p_of_noUpInTrainSets, p_of_noDownInTrainSets):
    noUpInNeighbors = 0
    noDownInNeighbors = 0
    for i in range(len(neighbors)):
        if neighbors[i][1] == 'up':
            noUpInNeighbors += 1
        else:
            noDownInNeighbors += 1

    p_of_noUpInNeighbors = noUpInNeighbors / len(neighbors)
    p_of_noDownInNeighbors = noDownInNeighbors / len(neighbors)

    joint_probability_up = p_of_noUpInTrainSets * p_of_noUpInNeighbors
    joint_probability_down = p_of_noDownInTrainSets * p_of_noDownInNeighbors

    if joint_probability_down > joint_probability_up:
        return 'down'
    else:
        return 'up'


def getAccuracy(y_test, predictions):
    correct = 0
    for x in range(len(y_test)):
        if y_test[x] == predictions[x]:
            correct += 1
    return (correct / float(len(y_test))) * 100.0


main()
