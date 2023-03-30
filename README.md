# Topic Classification

## What does it do?
- This code will train a model to classify the main topic in a body of text using training data provided by the user that already has the topic labeled and then predict the topics for a new set texts for which the topics have not been labeled. _Note that this code will take several minutes to run and will use n-2 cores on your computer (where n is the total number of available cores)._

## Setup
1. Fork or clone this repo to your own computer and `cd` into the directory
2. Make sure you have the proper dependencies installed. 
   -   You can do this by typing `pip install -r requirements.txt` in the command line
3. Download the SpaCy pipeline
    - `python -m spacy download en_core_web_sm`
4. Open the `input.yaml` file, update the inputs, and save the file (examples and descriptions of parameters below)
5. Run the main file by typing `python main.py` in the command line

## Input Parameter Descriptions
| Input Parameter | Description |
| --- | --- |
| training_data | User filepath to csv containing data on survey responses with the topics labeled in the data |
| prediction_data | User filepath to csv containing data on survey responses we want to predict the topics for |
| training_text | The name of the variable in the training_data that contains the text of the survey responses |
| training_labels | The name of the variable in the training_data that contains the labeled topic of the survey responses |
| prediction_text | The name of the variable in the prediction_data that contains the text of the survey responses |
| output_bigrams | User filepath to csv where you want to save the bigrams that are detected during text cleaning |
| output_predictions | User filepath to csv where you want to save the topic predictions _(this csv will be indentical to the prediction_data csv but with a new column containing the predicted topic for each survey response)_ |
| output_performance | User filepath to csv where you want to save the classification model accuracy _(this csv will contain 5 rows, with the percentage of topics accurately classified for 5 different subsets of the data. For example, if the first row says 30%, then the model accurately classified the topics for 30% of the survey responses in the first subset of the training data)_ |


## Example Input Customizations
```
---
# Data (must be a csv file)
training_data: C:/Users/emmea/Documents/GitHub/TopicClassification/data/training_data.csv
prediction_data: C:/Users/emmea/Documents/GitHub/TopicClassification/data/prediction_data.csv

# Name of variable containing text data 
training_text: text
training_labels: key_theme_1
prediction_text: text

# Output CSV (must be a csv file)
output_bigrams: C:/Users/emmea/Documents/GitHub/TopicClassification/data/bigrams.csv
output_predictions: C:/Users/emmea/Documents/GitHub/TopicClassification/data/topic_predictions.csv
output_performance: C:/Users/emmea/Documents/GitHub/TopicClassification/data/kfold_accuracy.csv
...
```

## Technical Details
None of the information in this section is necessary to know to be able to run this code, but we document each step in the code below:
### Text Cleaning and Tokenization
1. Basic Text Cleaning  
First we covert all characters to lowercase, remove punctuation, and remove stopwords (ex stopwords: are "and", "the", "from", etc)     
Before: `"I REALLY enjoyed learning college algebra from Professor Smith!!"`  
After: `"i really enjoyed learning college algebra professor smith"`

2. Lemmatization  
Next we lemmatize the text, meaning we convert all words to their root form (example: running and ran are converted to run)  
Before: `"i really enjoyed learning college algebra professor smith"`  
After: `"i really enjoy learn college algebra professor smith"`

3. Tokenization  
Now we take our clean text and convert it to a list of tokens  
Before: `"i really enjoy learn college algebra professor smith"`  
After: `["i", "really", "enjoy", "learn", "college", "algebra", "professor", "smith"]`

4. Lastly, we detect common bigrams (two word phrases) in the text and combine them.  
Before: `["i", "really", "enjoy", "learn", "college", "algebra", "professor", "smith"]`  
After: `["i", "really", "enjoy", "learn", "college_algebra", "professor", "smith"]`  

The detected bigrams are saved in a csv specified by the user in the YAML. The user may want to inspect the bigrams to make sure they look reasonable. The part of the code that trains a model to detect bigrams looks like this: `bigrams = Phrases(train_bigrams, min_count=10, threshold=25)`. It may be useful to play with the `min_count` (minimum number of times a bigram has to appear to be added to the list) and `threshold` (how sensitive the bigram detection model is with higher numbers indicating less sensative or fewer bigrams detected) depending on your data. You will get different bigrams list and potentially different final model accuracies if you change these parameters (though we do not expect final model accuracies to change very much).

### Converting the Text into Numeric Data
Before we can train a model, we need to convert our text into data. Suppose we have two survey responses. The first response says: `"I love all my classes"` and the second response says `"My classes were hard, but useful"`. After data cleaning, we create a matrix of word frequencies that look like the following:
|  | i | love | class | hard | useful |
| --- | --- | --- | --- | --- | --- |
| Response 1 | 1 | 1 | 1 | 0 | 0 |
| Response 2 | 0 | 0 | 1 | 1 | 1 |

Then we weight the frequencies by the number of time each word appears using term frequency–inverse document frequency (TF-IDF) to reflect how important a word is to the collection of survey responses.  

The last piece of cleaning we do to the text data is to oversample topics that don't appear very often so that the model does not consider them unimportant. We do this using a method called Synthetic Minority Over-sampling Technique (SMOTE).

### Training a Model: Hyperparameter Tuning
Now that we have cleaned our text and converted it to numeric data, we can train a model!
We choose to use train a Multinomial Naive Bayes classifier to classify the topic of our survey responses because this model performs well on small training sets. To train this model, we need to choose an optimal value for one of the model's parameters `alpha`. We choose the best value by running the model a bunch of times with different levels of `alpha` and seeing which value gives us the most accurate topic classifications (we explain how we calculate accuracy in the following section). This is the part of the code that takes the longest so we only test a couple values of alpha using Bayesian Optimization. In particular, we test 8 values at random, choose the best alpha out of those 8 random values and then test 4 more values near the best of the random 8. The part of the code that does this looks like this: `optimizer.maximize(init_points=8, n_iter=4)`. If you want to spend more time choosing the best value of `alpha` you can change this part of the code.


### Testing the Model
Lastly, we want to calculate how accurate the model is on "out-of-sample" data using K-folds Cross Validation. We choose K=5 which means we split the data into 5 subsamples. We combine 4 of the samples and train a model using this data, then we test how many of the topics we correctly classified on the 5th sample. We do this 5 times, training our model on a different set of 4 subsamples each time and testing the accuracy on the sample that is left out. We output the accuracy each time in a csv. Common practice is to report the average of these 5 accuracies as the model's overall accuracy.

### Classfiying Topics on New Data
After the training and testing our complete, we take a dataset of survey responses that has not been labeled and predict the topic for each response. We don't know how accurate the predictions are on this particular set of responses because we have not labeled the true topic.
