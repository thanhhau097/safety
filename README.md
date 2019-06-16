# Introduction

# Preprocessing data
In this data, we can see that with each bookingID, the sequence has many time steps, so the model can not learn the 
pattern easily.

To make model more efficient, there are some preprocessing steps that I used:
- Group by bookingID, sort by second field.
- Ignore bookingID, second fields: because they are not necessary.
- Ignore time steps that have large Accuracy field: because they are inaccurate.
- Ignore the last time steps that those velocity are equal to 0.
- Normalize each field of data into [0, 1]    
- Because of lacking data, I used a technique to generate more data:
    - With each sequence, I will generate a new sequence by getting one time step for each two time steps. For example:
        - Initiatial sequence: 0, 1, 2, 3, 4, 5, 6, 7
        - Some generated sequences:
            - 0, 2, 5, 6
            - 0, 3, 4, 6
            - 1, 2, 4, 7
            - ...
    - By doing this work, we can get more and more data to training. It made our model get higher accuracy.
    - In testing phase, to get more accurate result, with each initial sequence, we generate n sequences from original sequence,
    and predict all of them, then use voting to get final result. 

# Model
In this model, I use TCN architecture to train a sequence model. TCN can learn long sequence effectively.

This model have a light weight architecture, and it can predict very fast.

# Training
To train, you need to run training.py file, with parameter is path to data folder.

# Testing 
To test, you need to run testing.py file, with parameter is path to data folder.