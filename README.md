# quake-watch
Earthquakes can cause lots of destruction and harm which makes it one of the worst natural disasters in the world. Quake Watch's goal is to help bring awareness of when the next earthquake could possibly come so people can prepare for it. You might or might not know already but earthquakes can not be predicted. Quake Watch is not designed to be a completely accurate earthquake predicter but it estimates when the next earthquake will happen based on past data. Quake Watch analyzes past earthquakes and finds a pattern then it makes a prediction of when the next earthquake will strike. The analyzing and pattern finding is accomplished through machine learning algorithms.

```
random forest regressor
num_features ~ 17,500
Evaluation result on Test Data : mean squared error = 1.9729584493393746, score = 0.6813909079539409
```

# Notes:
- [X] Planning to use a different type of model since neural network won't predict.
- [X] Scale the data. https://www.kdnuggets.com/2018/11/keras-long-short-term-memory-lstm-model-predict-stock-prices.html
- [X] Test the new scaling code
- [X] Figure out why the training can be really bad. Inconsistent results
	- [X] Maybe add more LSTMs
	- [X] NOTE: the rnn wasn't able to make predictions based on a timestamp value
- [ ] Test on some user created data
- [ ] Options:
	- [X] tell users the magnitude of the next earthquake in their area (NOTE: not going with this idea)
	- [X] Try to fix model with timestamp values (NOTE: figured out it wasn't possible to get accurate results)
- [X] Remove hour time stamp (Just keep year, month, and day)
- [ ] Put model on a flask app or maybe use django
- [ ] Use google maps api
# Resources:
- Data and links:
 - http://service.scedc.caltech.edu/eq-catalogs/date_mag_loc.php
 - (Note now looking for earthquakes in CA)
 - http://scedc.caltech.edu/research-tools/downloads.html#catalogs (Download the SCEDC catalog)
 - https://developers.google.com/maps/documentation/javascript/examples/circle-simple (For drawing circles on maps radius measured in meters)
# Code:
```
# use for calculating the radius of an earthquake in meters
var radius = Math.sqrt(Math.pow(10, magnitude))/50000);
var radius = Math.pow(2, magnitude);
var radius = (Math.exp(magnitude/1.01-0.13))*1000;

 ```
# What I learned:
- In the beginning of this project, I believed I could create a machine learning model based on
past earthquakes that already had happened. I was hoping my model would be able to discover a 
pattern and give me a reasonable prediction of the next earthquake, but that was not the case. 
I learned that my model was overfitting and the model quickly learned to recognize the X_train 
dataset and output the exact answer that corresponded with each feature. My dataset was using 
a timestamp as a feature which didn’t help accelerate the learning process of my model. In 
fact, I don’t think the timestamp value had any effect on the learning of my model. The last 
reason why my model wasn’t working was the data wasn’t preprocessed properly.
- One of the most important problems to avoid in machine learning is overfitting. When testing 
the model I realized that the accuracy on the training data and the test data would be above 
ninety-eight percent. This realization was surprising and fantastic at first, but when I 
entered some of my own values into the model the outputs wasn’t what I expected. The values 
that were outputted where very close to each other. At first I thought it was a scaling issue 
and that the model was outputting the values based off the lowest and highest values, but the 
model still had the same relative values. The outputs with scaling didn’t match the values 
that of the outputs of the test data at all. 
- Choosing the data is almost as important as choosing the right model. In my dataset, I have a 
date represented as a timestamp and I feed it into the timestamp. Through extensive testing I
learned that the timestamp didn’t have much effect on the training. I would enter a timestamp 
number and run it into the model. The model would output the same value every time meaning 
there must be something wrong with the data. I kept experimenting and I found out that the 
timestamp variable had no effect on the models output.
- I figured out the method one uses to feed into the model can have a major effect on its 
training. I was searching online for resources and tutorials and I always came across the 
method where the programmer creates a lookback function. I realized this was how the model 
could predict things in the future. I could’ve done the same but my data could possibly have 
two earthquakes in the same day and time which would be extremely difficult for the model and 
me to figure out how to deal with it. 
- From this learning experience I learned much about machine learning and what to do, but mainly
what not to do. I figured out the importance of finding or making a dataset with the correct 
features and how to preprocess them correctly. Learning about and experiencing overfitting is
always a good learning lesson. What I’m probably going to do now is run the dataset through a 
linear regression model and put the model on a live website.
