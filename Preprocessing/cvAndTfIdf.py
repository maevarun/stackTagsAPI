from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import gensim
import joblib
import spacy
import pickle


def sent_to_words(sentence):
    yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
    return sentence

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append(" ".join(
            [token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
    return texts_out

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

vectorizer = CountVectorizer(analyzer='word',
                         min_df=10,                        # minimum reqd occurences of a word
                         stop_words='english',             # remove stop words
                         lowercase=True,                   # convert all words to lowercase
                         token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3
                         max_features=50000             # max number of uniq words
                        )

# def vectorisation(question) :
#     vectorizer = CountVectorizer(analyzer='word',
#                              min_df=10,                        # minimum reqd occurences of a word
#                              stop_words='english',             # remove stop words
#                              lowercase=True,                   # convert all words to lowercase
#                              token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3
#                              max_features=50000             # max number of uniq words
#                             )
#
#     question_vec = vectorizer.fit_transform(question)
#     return question_vec
#     # save the vectorizer
#     pickle.dump(vectorisation, open('vec.pkl', 'wb'))


# vectorizer = CountVectorizer()
#
# # save the vectorizer
# pickle.dump(vectorizer, open('vec.pkl', 'wb'))

# example for saving python object as pkl
# joblib.dump(vectorizer, "vectorizer.pkl")
# joblib.dump(vectorizer, "vectorizer.pkl")