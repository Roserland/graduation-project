import pandas as pd
import numpy as np

origin = pd.read_csv('./coords/coords.csv')
typeList_1 = ['LUSC', 'LUAD']   # 54G, 181G
typeList_1 = ['UCS', 'UCEC']    #7.7G, 71G
typeList_1 = ['KICH', 'KIRC', 'KIRP']  # 85G, 163G, 114G

rate = [3, 10]

def divide(origin, typeList, RATE):
    pass

def split_train_test(data, train_ratio = 0.7):
    length = len(data)

    train_len = int(length * train_ratio)
    test_len  = length - train_len

    train = data[:train_len]
    test  = data[train_len:]

    return train, test

if __name__ == '__main__':
    # KICH = origin[origin['TypeName'] == 'KICH']
    # KIRC = origin[origin['TypeName'] == 'KIRC']
    # KIRP = origin[origin['TypeName'] == 'KIRP']
    #
    # print(len(KICH), len(KIRC), len(KIRP))
    #
    # KICH = KICH[:]
    # KIRC = KIRC[:150]
    # KIRP = KIRP[:150]
    #
    # KICH_train, KICH_test = split_train_test(KICH)
    # KIRC_train, KIRC_test = split_train_test(KIRC)
    # KIRP_train, KIRP_test = split_train_test(KIRP)

    LUSC = origin[origin['TypeName'] == 'LUSC']
    LUAD = origin[origin['TypeName'] == 'LUAD']

    print(len(LUSC), len(LUAD))

    LUSC = LUSC[:]
    LUAD = LUAD[:240]

    LUSC_train, LUSC_test = split_train_test(LUSC)
    LUAD_train, LUAD_test = split_train_test(LUAD)

    train = LUSC_train.append(LUAD_train)
    test  = LUSC_test.append(LUAD_test)

    print(len(train), len(test))
    print(train.head(4))

    train.to_csv('./coords/LU_TwoTypes_Train.csv', index = None)
    test.to_csv('./coords/LU_TwoTypes_Test.csv', index = None)







