

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import Levenshtein as lev
import spacy
from sentence_transformers import SentenceTransformer, util

from similarity.kl import score
import time


def cosin_text(text1, text2):

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])

    cos_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    print("Cosine Similarity:", cos_sim[0][0])


#defined as the size of the intersection divided by the size of the union of two label sets
def jaccard_text(text1, text2):

    a = set(text1.split())
    b = set(text2.split())
    c = a.intersection(b)
    similarity = float(len(c)) / (len(a) + len(b) - len(c))
    print("Jaccard Similarity:", similarity)

#measures the minimum number of edits (insertions, deletions, substitutions) required to change one string into the other.

def levenshtein_text(text1, text2):
    distance = lev.distance(text1, text2)
    similarity = 1 - (distance / max(len(text1), len(text2)))
    print("Levenshtein Similarity:", similarity)


#utilizes word vectors to find the similarity between texts.
#This method uses pre-trained word vectors and can provide a more semantically aware similarity
def spacy_text(text1, text2):
    # Load pre-trained word vectors model
    nlp = spacy.load('en_core_web_md')

    doc1 = nlp(text1)
    doc2 = nlp(text2)

    # Compute similarity
    similarity = doc1.similarity(doc2)
    print("Spacy Word Vectors Similarity:", similarity)




def bert_embedding_text(text1, text2):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Get the embeddings
    embeddings1 = model.encode(text1, convert_to_tensor=True)
    embeddings2 = model.encode(text2, convert_to_tensor=True)
    # Compute similarity
    similarity = util.pytorch_cos_sim(embeddings1, embeddings2)
    print("BERT Embeddings Similarity:", similarity.item())



def number_check(number1, number2, threshold = 0, error_rate=0):
    difference = abs(number1-number2)
    #Relative error is a measure of the discrepancy between an observed (approximate) value and the true (exact) value, relative to the true value.
    relative_error = difference/ max(abs(number1), abs(number2))

    # return difference check, relative error check
    return difference < threshold, relative_error < error_rate







'''
#text1 = "Hello, how are you?"
#text2 = "Hello, how have you been?"

text1 = 'The insightful presentation captivated the attentive audience'
text2 = 'The attentive audience was engrossed by the insightful presentation'


a = time.time()
cosin_text(text1, text2)
b = time.time()
print('Time cost: ', abs(a-b))
jaccard_text(text1, text2)
a = time.time()
print('Time cost: ', abs(a-b))
levenshtein_text(text1, text2)
b = time.time()
print('Time cost: ', abs(a-b))
spacy_text(text1, text2)
a = time.time()
print('Time cost: ', abs(a-b))
bert_embedding_text(text1, text2)
b = time.time()
print('Time cost: ', abs(a-b))
score(text1, text2)
a = time.time()
print('Time cost: ', abs(a-b))

'''