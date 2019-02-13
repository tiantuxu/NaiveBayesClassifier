## NaiveBayesClassifier
### CS573 Data Mining Assignments 2

### Question 1
```
$ python preprocess.py dating-full.csv dating.csv
```

### Question 2
```
$ python 2_1.py
$ python 2_2.py
```

### Question 3
```
$ python discretize.py dating.csv dating-binned.csv
```
### Question 4
Split the data to training and test sets.
```
$ python split.py
```

### Question 5
```
$ python 5_1.py
```
The script calls nbc(t_frac).
```
$ python 5_2.py
```
The script calls nbc(t_frac, df_train), with training data for each bin value discretizes on the fly and passes to the nbc training function.
```
$ python 5_3.py
```
The script calls nbc(t_frac) for each f.
