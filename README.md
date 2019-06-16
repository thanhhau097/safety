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

# Model
In this model, I use TCN architecture to train a sequence model. TCN can learn long sequence effectively.


# Training
To train, you need to run training.py file, with parameter is path to data folder.

# Testing 
To test, you need to run testing.py file, with parameter is path to data folder.