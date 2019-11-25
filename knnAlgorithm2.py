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
    getStockDataFromFile(fileName, companyStockName, startDate, endDate)

    print("Predicting for", companyStockName)
    print("Training set: " + repr(len(trainingSet)))
    print("Test set: " + repr(len(testSet)))
    totalSet += len(trainingSet) + len(testSet)
    print("Total: " + repr(totalSet))


def getStockDataFromFile(fileName, companyStockName, startDate, endDate):
    stockData = dr.DataReader(companyStockName, 'yahoo', startDate, endDate)
    stck_json = stockData.to_json(orient="index", date_format='iso')
    stck_dates = json.loads(stck_json)
    plt.plot(stockData["Adj Close"])
    plt.title("Stock movement of " + companyStockName)


main()
