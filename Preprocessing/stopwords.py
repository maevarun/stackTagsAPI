import nltk
import re
import string
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('words')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Create function to remove html tags
def remove_html(text):
    html = re.compile(r"<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
    return re.sub(html, "", text)

punct = string.punctuation

def kill_punctuation(word):
    for c in punct:
        word = word.replace(c, "")
    return word

stopwords = nltk.corpus.stopwords.words('english') + ["V","v", "trying", "tried", "work", "want", "also", "know", "see", "make", "need", "dont", "difference", "using"]
words = set(nltk.corpus.words.words())
# tokenizer = nltk.RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()

def Preprocess_listofSentence(sentence):
    sentence_w_punct = kill_punctuation(sentence)
    sentence_w_num = ''.join(i for i in sentence_w_punct if not i.isdigit())
    tokenize_sentence = nltk.tokenize.word_tokenize(sentence_w_num)
    # tokenize_sentence = tokenizer.tokenize(sentence_w_num)
    words_w_stopwords = [i for i in tokenize_sentence if i not in stopwords]
    words_lemmatize = (lemmatizer.lemmatize(w) for w in words_w_stopwords)
    sentence_clean = ' '.join(w for w in words_lemmatize if w.lower() in words or not w.isalpha())
    return sentence_clean
