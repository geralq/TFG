import os
import pandas as pd
import random

def split_data(input_dir):

    trainl = []

    testl = []

    vall = []

    for subfolder in os.listdir(input_dir):
        subfolder_path = os.path.join(input_dir, subfolder)
    
        files = [f for f in os.listdir(subfolder_path)]

        random.shuffle(files)

        length = len(files)

        train = length * 0.7
        test = length * 0.15
        val = length * 0.15

        for i, file in enumerate(files):
            if i < train:
                trainl.append(f"{subfolder}/{file}")
            elif i < train + val:
                vall.append(f"{subfolder}/{file}")
            elif i < train + val + test:
                testl.append(f"{subfolder}/{file}")
    
    df = pd.DataFrame(trainl, columns=['train'])
    df.to_csv('/home/gerardo/LSE_HEALTH/LSE_TFG/train_test_val_split/train_weigthed_samples.csv', index=False)

    df = pd.DataFrame(testl, columns=['test'])
    df.to_csv('/home/gerardo/LSE_HEALTH/LSE_TFG/train_test_val_split/test_weigthed_samples.csv', index=False)

    df = pd.DataFrame(vall, columns=['val'])
    df.to_csv('/home/gerardo/LSE_HEALTH/LSE_TFG/train_test_val_split/val_weigthed_samples.csv', index=False)


        