#!/usr/bin/env python
from gensim import corpora, models, similarities
from nltk.tokenize import TweetTokenizer
import nltk.stem
import math

def preprocess_docs(documents):
    tokenizer = TweetTokenizer()
    english_stemmer = nltk.stem.SnowballStemmer('english')

    texts = [tokenizer.tokenize(d) for d in documents]

    stemmed_texts = []
    for text in texts:
        stemmed_text = [english_stemmer.stem(t) for t in text]
        stemmed_texts.append(stemmed_text)
    return stemmed_texts

def strong_similarities_and_appropriate_links_thresh(lsi_queries, index):
    '''
    Returns a similarity dictionary with all the sentences
    in lsi_queries, and their lists of strongest links tuples
    with the sentence id link and the similarity percentage.
    '''
    total_links = 0
    similarity_dict = {}

    for i, query in enumerate(lsi_queries):
        sims = index[query]

        strong_sims = [s for s in list(enumerate(sims)) if s[1] > 0.999]

        similarity_dict[i] = strong_sims
        links = len(strong_sims)

        total_links += links

    # max_links is the average number of links per query sentence
    min_links = 1
    max_links = math.ceil(total_links/float(len(lsi_queries)))
    thresh = (min_links, max_links) # non-inclusive
    return similarity_dict, thresh


def perform_queries_and_get_links(lsi_queries, index):
    s_dict, thresh = strong_similarities_and_appropriate_links_thresh(lsi_queries,
                                                                      index)
    pruned_dict = {sid: simils for sid, simils in zip(s_dict.keys(), s_dict.values())
                   if len(simils) > thresh[0] and len(simils) < thresh[1]}

    strong_sentences = len(pruned_dict.keys())
    selected_pairs = sum([len(x) for x in pruned_dict.values()])

    #print "\n{0} strong sentences".format(strong_sentences)
    #print "{0} total sentence-sentence pairs".format(selected_pairs)
    print thresh
    return pruned_dict

def find_links_between_in(documents, comments_sentences):
    '''
    params:
        documents = all sentences (article and comments alike) in order
        comments_sentences = all comments_sentences in order

    return:
        dictionary with the comment sentence number (not its sentence id)
        and the sentence id of the sentences it is linked to.

    The caller might want to post-process the dictionary, so
    that the dictionary's keys are actually the comment's sentence
    ids.

    Possible improvement of this function can be done, by asking for another
    array as a parameter that will be used as a semantic corpus. This array
    can be obtained by using the keywords in the guardian article (that are
    part of the meta data of the site) to look for more articles, tokenize
    their sentences and vecotrize them to generate a richer semantic corpus.
    '''
    stemmed_texts = preprocess_docs(documents)
    dictionary = corpora.Dictionary(stemmed_texts)
    dictionary.filter_extremes(no_below=1, no_above=0.8)
    corpus = [dictionary.doc2bow(text) for text in stemmed_texts]

    lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)
    index = similarities.MatrixSimilarity(lsi[corpus])

    stemmed_queries = preprocess_docs(comments_sentences)
    query_dict = corpora.Dictionary(stemmed_queries)
    lsi_queries = [lsi[query_dict.doc2bow(text)] for text in stemmed_queries]
    similarity_dict = perform_queries_and_get_links(lsi_queries, index)
    return similarity_dict
