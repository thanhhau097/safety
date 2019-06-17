# Introduction
### PROBLEM STATEMENT

Given the telematics data for each trip and the label if the trip is tagged as dangerous driving, 
derive a model that can detect dangerous driving trips.

### Dataset
The given dataset contains telematics data during trips (bookingID).
 Each trip will be assigned with label 1 or 0 in a separate label file to indicate dangerous driving. 
 Pls take note that dangerous drivings are labelled per trip, while each trip could contain thousands of telematics data points.
 participants are supposed to create the features based on the telematics data before training models.

| Field          | Description                             |
|----------------|-----------------------------------------|
| bookingID      | trip id                                 |
| Accuracy       | accuracy inferred by GPS in meters      |
| Bearing        | GPS bearing in degree                   |
| acceleration_x | accelerometer reading at x axis (m/s2)  |
| acceleration_y | accelerometer reading at y axis (m/s2)  |
| acceleration_z | accelerometer reading at z axis (m/s2)  |
| gyro_x         | gyroscope reading in x axis (rad/s)     |
| gyro_y         | gyroscope reading in y axis (rad/s)     |
| gyro_z         | gyroscope reading in z axis (rad/s)     |
| second         | time of the record by number of seconds |
| Speed          | speed measured by GPS in m/s            |

# Preprocessing data
In this data, we can see that with each bookingID, the sequence has many time steps, so the model can not learn the 
pattern easily.

To make model more efficient, there are some preprocessing steps that I used:
- *Group by bookingID, sort by second field*.
- *Ignore bookingID, second fields*: because they are not necessary.
- *Ignore time steps that have large Accuracy value*: because they are inaccurate.
- *Ignore the last time steps that those velocity are equal to 0*: because those time steps have no meaning.
- *Normalize each field of data into [0, 1]*: It will make our model more easier in training and boost our model accuracy.
- Because of lacking data, I used a technique to **generate more data**:
    - With each sequence, I will generate a new sequence by getting one time step for each two time steps. For example:
        - Initiatial sequence: 0, 1, 2, 3, 4, 5, 6, 7
        - Some generated sequences:
            - 0, 2, 5, 6
            - 0, 3, 4, 6
            - 1, 2, 4, 7
            - ...
    - By doing this work, we can get **more and more data for training**. It made our model get **higher accuracy**.
    - In testing phase, to get more accurate result, with each initial sequence, we **generate n sequences from original sequence**,
    and predict all of them, then use **voting** to get final result. 

# Model
In this model, I use **TCN** architecture to train a sequence model. TCN can learn long sequence effectively.
- TCNs exhibit longer memory than recurrent architectures with the same capacity.
- Constantly performs better than LSTM/GRU architectures on a vast range of tasks (Seq. MNIST, Adding Problem, Copy Memory, Word-level PTB...).
- Parallelism, flexible receptive field size, stable gradients, low memory requirements for training, variable length inputs...

This model have a light weight architecture, and it can predict very fast.

![](tcn.png)

# Training
To train, you need to run training.py file, with parameter is path to data folder.

```buildoutcfg
python training.py
```

Folder structure:
```buildoutcfg
.
+-- features
|   +-- features_1.csv
|   +-- features_2.csv
|   +-- features_3.csv
|   +-- features_4.csv
+-- labels
|   +-- labels.csv
```

# Testing 
To test, you need to run testing.py file, with parameter is path to data folder.

```buildoutcfg
python testing.py
```

This data folder have same structure with training data folder.
```buildoutcfg
.
+-- features
|   +-- features_1.csv
|   +-- features_2.csv
|   +-- features_3.csv
|   +-- features_4.csv
+-- labels
|   +-- labels.csv
```

This method will give us a csv file, which is similar to labels file.