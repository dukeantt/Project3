import datetime


def main():
    startDate = datetime.datetime(2014, 1, 1)
    endDate = datetime.date.today()
    predict(10, 'AppleHistoricalQuotes', "APPL", startDate, endDate)


# k = so hang xom
def predict(k, fileName, company, startDate, endDate):
    attributes = ["date", "close", "volume", "open", "high", "low"]
    trainingSet = []
    testSet = []
