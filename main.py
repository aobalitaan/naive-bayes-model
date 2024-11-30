import random
import csv

import naiveBayes

TARGET = 'Scholarship holder'
METRIC = 'accuracy'     # metric where the mode select/base which is the "best", maximized

# Set to continuous (0-200), normalized to (0-20) for now
FIX1 = 'Admission grade'
FIX2 = 'Previous qualification (grade)'
# Rounded to nearest 0.5
FIX3 = 'GDP'
FIX4 = 'Inflation rate'
FIX5 = 'Unemployment rate'

def load_dataset(file_path):
    dataset = []

    with open(file_path, mode='r', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file, delimiter=';')

        for row in reader:
            row[FIX1] = str(float(row[FIX1])//10)
            row[FIX2] = str(float(row[FIX2])//10)

            row[FIX3] = str(round(float(row[FIX3])*2)/2)
            row[FIX4] = str(round(float(row[FIX4])*2)/2)
            row[FIX5] = str(round(float(row[FIX5])*2)/2)

            dataset.append(row)
    
    return list(dataset[0].keys()), dataset

def split_dataset(dataset):     # Splits the dataset
    wtarget = [item for item in dataset if item[TARGET] == '1']     # Gets those with target = 1
    wotarget = [item for item in dataset if item[TARGET] == '0']    # Gets those with target = 0

    random.shuffle(wtarget)     # Shuffles them
    random.shuffle(wotarget)


    # Checks if imbalance
    curr_ratio = len(wtarget) / (len(wtarget) + len(wotarget))  

    
    if curr_ratio < 0.4:    # If [target==1] < [target==0] , normalizes it to within either 60:40 or 40:60
        max_wotarget = int(len(wtarget) * 60 / 40)
        wotarget = wotarget[:max_wotarget]
    elif curr_ratio > 0.6:
        max_wtarget = int(len(wotarget) * 60 / 40)
        wtarget = wtarget[:max_wtarget]

    # For now 80% training, 20% testing
    training = {
        'wtarget' : wtarget[0:int(len(wtarget)*.8)],
        'wotarget' : wotarget[0:int(len(wotarget)*.8)]
    }

    testing = {
        'wtarget' : wtarget[int(len(wtarget)*.8):len(wtarget)],
        'wotarget' : wotarget[int(len(wotarget)*.8):len(wotarget)]
    }

    return testing, training

def main():
    varList, dataset = load_dataset('data.csv')     # Loads the dataset
    
    varList.remove(TARGET)  # Removes the target from list of variables to test
    varList.remove("Status")    # Removes status since this will not be known yet at the beginning of an academic year

    testing, training = split_dataset(dataset)      # Splits the dataset, no validation set yet

    naiveBayes.main(varList, training, testing, TARGET, METRIC)   



if __name__ == "__main__":
    main()

