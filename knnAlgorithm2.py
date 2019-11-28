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
    print("Training set: " + repr(len(y_train[0])))
    print("Test set: " + repr(len(y_test[0])))
    totalSet += len(trainingSet) + len(testSet)
    print("Total: " + repr(totalSet))

    X_train = X_train[0]
    y_train = y_train[0]
    X_test = X_test[0]
    y_test = y_test[0]
    predictAndGetAccuracy(X_train, y_train, X_test, y_test, k, companyStockName)

    # predictAndGetAccuracy(trainingSet,testSet,k,companyStockName)


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
        # minus 1 because we are predicting for next day
        # split stock data set to training set and test set
        for x in range(len(dataset) - 1):
            # convert the content to float
            # minus 1 because last is string for up or down
            for y in range(1, len(attributes) - 1):
                dataset[x][y] = float(dataset[x][y])
            stock_y.append(dataset[x][-1])
            stock_X.append(dataset[x][: len(dataset[x]) - 1])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])

        Xtrain, Xtest, ytrain, ytest = train_test_split(stock_X, stock_y, test_size=0.33)

        X_train.append(Xtrain)
        y_train.append(ytrain)
        X_test.append(Xtest)
        y_test.append(ytest)


def predictAndGetAccuracy(X_train, y_train, X_test, y_test, k, companyStockName):
    predictions = []
    for x in range(len(y_test)):
        neighbors = getNeighbors(X_train, y_train, X_test[x], y_test[x], k)
        result = getResponse(neighbors)
        predictions.append(result)

    accuracy = getAccuracy(y_test, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')


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


def getAccuracy(y_test, predictions):
    correct = 0
    for x in range(len(y_test)):
        if y_test[x] == predictions[x]:
            correct += 1
    return (correct / float(len(y_test))) * 100.0


main()
