
#-------------------------------------------------------
# import primary packages
import yaml
import pandas as pd
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import spacy
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from bayes_opt import BayesianOptimization
from gensim.models import Phrases


# load inputs from YAML
with open("../input.yaml", 'r') as input_file: 
    inputs = yaml.safe_load(input_file)



#-------------------------------------------------------
# open and format datasets

# open datasets
df = pd.read_csv(inputs["training_data"])
predict_df = pd.read_csv(inputs["prediction_data"])

# drop NA values and reset index
df = df.dropna(subset = [inputs["training_text"], inputs["training_labels"]]).reset_index(drop=True)
predict_df = predict_df.dropna(subset = [inputs["prediction_text"]]).reset_index(drop=True)

# divide training set into text and labels
x, y = df[inputs["training_text"]], df[inputs["training_labels"]]



#-------------------------------------------------------
# prepare text cleaning models/functions

#python3 -m spacy download en_core_web_sm in console in first run
nlp = spacy.load('en_core_web_sm', disable = ['parser','ner'])

# define cleaning and tokenizing function
def clean_tokens(text, lower=True, lemmatize=True, rm_punc=True, rm_stopwords=True, spacy_pipeline=nlp):
    text = [str(text)]

    # Load stopwords
    stop_words = spacy_pipeline.Defaults.stop_words

    # Convert text to series of docs
    docs = list(spacy_pipeline.pipe(text))

    # Initialize list of sentences
    text = []

    for doc in docs:

        # Tokenize and Clean Text
        for token in doc:

            # Remove Punctuation
            if rm_punc and token.is_punct: continue

            # Remove Stopwords
            if rm_stopwords and token.text in stop_words: continue

            # Lemmatize
            if lemmatize: t = token.lemma_
            else: t = token.text

            # Lowercase
            if lower: t = t.lower()

            # Add token to sentence
            text.append(t)
            
    return text

# train bigrams model
train_bigrams = [clean_tokens(i) for i in x]
bigrams = Phrases(train_bigrams, min_count=10, threshold=25)

# Output list of bigrams
bigrams_list = [ngrams.decode('utf-8') for ngrams, _ in bigrams.vocab.items() if '_' in ngrams.decode('utf-8')]
pd.DataFrame(bigrams_list).to_csv(inputs["output_bigrams"], index=False, na_rep='NA', header=['bigrams'])

# create function to implement n_grams model on clean text
def clean_n_grams(text, n_grams_model=bigrams):
    tokens = clean_tokens(text)
    return n_grams_model[tokens]



#-------------------------------------------------------
# tune hyperparameter

def objective_function(alph, x=x, y=y):

    cv = StratifiedShuffleSplit(n_splits=3, test_size=0.3, random_state=round(alph*1000))  

    model = Pipeline([
    ('vect', CountVectorizer(tokenizer=clean_n_grams)),
    ('tfidf', TfidfTransformer()),
    ('smote', SMOTE(random_state=round(alph*1000))),
    ('mnb', MultinomialNB(alpha = alph))
    ])
    
    scores = cross_val_score(model, x, y, scoring='accuracy', cv=cv, n_jobs=-2)
    mean = sum(scores) / len(scores)
    return mean

# numerical optimization using bayesian optimization
optimizer = BayesianOptimization(f=objective_function, pbounds={'alph': (0, 1)}, random_state=13)
optimizer.maximize(init_points=8, n_iter=4)

# Output best hyperparater
bestalpha = optimizer.max['params']['alph']



#-------------------------------------------------------
# Build textclassifier pipeline using best hyperparameter

tunedtextclassifier = Pipeline([
    ('vect', CountVectorizer(tokenizer=clean_n_grams)),
    ('tfidf', TfidfTransformer()),
    ('smote', SMOTE(random_state=13)),
    ('mnb', MultinomialNB(alpha = bestalpha))
])


#-------------------------------------------------------
# cross-validate model

# define straified k-folding approach with more splits
skf = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=13)
metrics = cross_val_score(tunedtextclassifier, x, y, scoring='accuracy', cv=skf, n_jobs=-2)

# save accuracy scores
pd.DataFrame(metrics).to_csv(inputs["output_performance"], na_rep='NA', header=['% Accuracy of Predictions on 10 Subsamples'])


#-------------------------------------------------------
# create final model predictions

# fit model to entire training dataset
tunedtextclassifier.fit(x, y)

# create final predictions
predict_df['pred_label'] = tunedtextclassifier.predict(predict_df[inputs["prediction_text"]])

# save predictions
predict_df.to_csv(inputs["output_predictions"], index=False, na_rep='NA')
