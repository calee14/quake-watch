# quake-watch
Earthquakes can cause lots of destruction and harm which makes it one of the worst natural disasters in the world. Quake Watch's goal is to help bring awareness of when the next earthquake could possibly come so people can prepare for it. You might or might not know already but earthquakes can not be predicted. Quake Watch is not designed to be a completely accurate earthquake predicter but it estimates when the next earthquake will happen based on past data. Quake Watch analyzes past earthquakes and finds a pattern then it makes a prediction of when the next earthquake will strike. The analyzing and pattern finding is accomplished through machine learning algorithms.

```
recurrent neural network model
epochs = 10
num_features ~ 17,500
neurons = 16
Evaluation result on Test Data : Loss = 0.0035595247093346324, accuracy = 0.9815531769528582
```

# Notes:
- [X] Planning to use a different type of model since neural network won't predict.
- [X] Scale the data. https://www.kdnuggets.com/2018/11/keras-long-short-term-memory-lstm-model-predict-stock-prices.html
- [ ] Test the new scaling code
- [ ] Figure out why the training can be really bad. Inconsistent results
	- [ ] Maybe add more LSTMs
- [ ] Test on some user created data
- Options:
 - [ ] tell users the magnitude of the next earthquake in their area
 - [ ] Try to fix model with timestamp values
- [ ] Remove hour time stamp (Just keep year, month, and day)
# Resources:
- Data:
 - http://service.scedc.caltech.edu/eq-catalogs/date_mag_loc.php
 - (Note now looking for earthquakes in CA)
 - http://scedc.caltech.edu/research-tools/downloads.html#catalogs (Download the SCEDC catalog)
