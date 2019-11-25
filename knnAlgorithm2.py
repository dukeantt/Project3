import csv
import datetime
import pandas_datareader.data as dr
import pandas
import matplotlib.pyplot as plt
import json


def main():
    startDate = datetime.datetime(2014, 1, 1)
    endDate = datetime.date.today()
    # predict stock tendance of Amazon
    predict(10, 'AmazonHistoricalQuotes.csv', "AMZN", startDate, endDate)


# k = so hang xom
def predict(k, fileName, companyStockName, startDate, endDate):
    attributes = ["date", "close", "volume", "open", "high", "low"]
    trainingSet = []
    testSet = []
    totalSet = 0
    getStockDataAndWriteToFile(fileName, companyStockName, startDate, endDate)

    print("Predicting for", companyStockName)
    print("Training set: " + repr(len(trainingSet)))
    print("Test set: " + repr(len(testSet)))
    totalSet += len(trainingSet) + len(testSet)
    print("Total: " + repr(totalSet))


def getStockDataAndWriteToFile(fileName, companyStockName, startDate, endDate):
    stockData = dr.DataReader(companyStockName, 'yahoo', startDate, endDate)
    stockJson = stockData.to_json(orient="index", date_format='iso')
    stockDates = json.loads(stockJson)

    # plt.plot(stockData["Adj Close"])
    # plt.title("Stock movement of " + companyStockName)

    with open(fileName, 'w', newline='') as stockFile:
        stockWriter = csv.writer(stockFile)
        sortedDate = sorted(stockDates.keys())
        for i in sortedDate:
            formatDate = i[:10]
            prevClose = stockDates[i]["Adj Close"]
            stockWriter.writerow(
                [formatDate] + [stockDates[i]["Open"]] + [stockDates[i]["High"]] + [stockDates[i]["Low"]] + [
                    stockDates[i]["Adj Close"]] + [change(stockDates[i]["Adj Close"], prevClose)])


def change(today, yesterday):
    if today > yesterday:
        return 'up'
    return 'down'


main()
