import pandas as pd


def show_silscores(silscores):
    dfscores = pd.DataFrame(silscores, index = [0])
    print(dfscores)