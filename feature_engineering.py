#!/bin/bash
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import dump_svmlight_file

def features_addition(data):
    print("helo")
    return data

def features_remotion(data):
    data = data.drop(["date_time", "gross_bookings_usd"], axis=1)
    return data



def main():
    path="C://Users//david\Desktop//VU amsterdam//Data mining"
    train = pd.read_csv(path+"/training_set_VU_DM.csv", nrows=1000000)
    new_train = features_remotion(train)
    new_train = features_addition(new_train)


if __name__ == '__main__':
    main()
