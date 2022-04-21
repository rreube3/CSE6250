import pandas as pd
import numpy as np
import statistics
import spacy
from spacy.matcher import Matcher
from spacy.util import filter_spans
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# Add raw data files as pandas dataframes using read_csv functionality
raw_data = pd.read_csv("data/500_Reddit_users_posts_labels.csv")
afinn_data = pd.read_csv('data/AFINN-en-165.txt', sep="\t", header=None)
labMT_data = pd.read_csv("data/labMT")

# Convert dataframes to dictionaries of terms with corresponding values
dict_afinn = dict(zip(afinn_data[0], afinn_data[1]))
dict_hrank = dict(zip(labMT_data["word"], labMT_data["happiness_rank"]))
dict_havg = dict(zip(labMT_data["word"], labMT_data["happiness_average"]))
dict_hstdv = dict(zip(labMT_data["word"], labMT_data["happiness_standard_deviation"]))
dict_twit = dict(zip(labMT_data["word"], labMT_data["twitter_rank"]))
dict_goog = dict(zip(labMT_data["word"], labMT_data["google_rank"]))
dict_nyt = dict(zip(labMT_data["word"], labMT_data["nyt_rank"]))
dict_lyr = dict(zip(labMT_data["word"], labMT_data["lyrics_rank"]))

def clean_data(post):
    # create cleaned dataset to be used for generating scores
    lower_post = post.lower()
    # remove punctuation from post
    exclude = set(",.:;'\"-?!/")
    exclude_post = "".join([(char if char not in exclude else " ") for char in lower_post])
    # ensure only lowercase letters and whitespace are in the final post
    whitelist = set('abcdefghijklmnopqrstuvwxyz ')
    filter_post = ''.join(filter(whitelist.__contains__, exclude_post))
    return filter_post

# loop through dataset posts and get aggregate "positivity" scores, store them in a new df column
def afinn_score(post):
    cleaned_data = clean_data(post)
    # return mean AFINN score for post
    score_list = [dict_afinn[word] for word in cleaned_data.split() if word in dict_afinn.keys()]
    if len(score_list) > 0:
        return statistics.mean(score_list)
    return 0

# calculate first person pronoun ratio
def fpp_ratio_score(post):
    cleaned_data = clean_data(post)
    # create a list of first person and other (secondary/tertiary) pronouns
    fpp = ["i", "me", "my", "mine", "we", "us", "our", "ours"]
    opp = ["you", "your", "yours", "he", "she", "him", "her", "his", "hers", "they", "them", "their", "theirs", "youll", "youd"]
    # calculate ratio of first person pronouns used in post to other pronouns (use 0.1 to avoid divide by 0 error for "other" pronouns)
    first, other = 0, 0.1
    for word in cleaned_data.split():
        if word in fpp:
            first += 1
        elif word in opp:
            other += 1
    return first/other

def hrank_score(post):
    cleaned_data = clean_data(post)
    # return average happiness rank score for post
    return statistics.mean([dict_hrank[word] for word in cleaned_data.split() if word in dict_hrank.keys()])

def havg_score(post):
    cleaned_data = clean_data(post)
    # return average happiness avg score for post
    return statistics.mean([dict_havg[word] for word in cleaned_data.split() if word in dict_havg.keys()])

def hstdv_score(post):
    cleaned_data = clean_data(post)
    # return average happiness stdev score for post
    return statistics.mean([dict_hstdv[word] for word in cleaned_data.split() if word in dict_hstdv.keys()])

def twit_score(post):
    cleaned_data = clean_data(post)
    # return average twitter rank score for post
    word_scores = []
    for word in cleaned_data.split():
        # check if word in dictionary
        if word in dict_twit.keys():
            # check if word value is not null
            if dict_twit[word] == dict_twit[word]:
                word_scores.append(dict_twit[word])
    return statistics.mean(word_scores)

def goog_score(post):
    cleaned_data = clean_data(post)
    # return average google rank score for post
    word_scores = []
    for word in cleaned_data.split():
        # check if word in dictionary
        if word in dict_goog.keys():
            # check if word value is null
            if dict_goog[word] == dict_goog[word]:
                word_scores.append(dict_goog[word])
    return statistics.mean(word_scores)

def nyt_score(post):
    cleaned_data = clean_data(post)
    # return average nyt rank score for post
    word_scores = []
    for word in cleaned_data.split():
        # check if word in dictionary
        if word in dict_nyt.keys():
            # check if word value is null
            if dict_nyt[word] == dict_nyt[word]:
                word_scores.append(dict_nyt[word])
    return statistics.mean(word_scores)

def lyric_score(post):
    cleaned_data = clean_data(post)
    # return average lyric rank score for post
    word_scores = []
    for word in cleaned_data.split():
        # check if word in dictionary
        if word in dict_lyr.keys():
            # check if word value is null
            if dict_lyr[word] == dict_lyr[word]:
                word_scores.append(dict_lyr[word])
    return statistics.mean(word_scores)

def avg_tree_depth(post):
    # Computes avg parse tree height for a series of sentences
    # NOTE: I referenced source code at https://gist.github.com/drussellmrichie/47deb429350e2e99ffb3272ab6ab216a
    # to perform this calculation.
    # NOTE: You must download the spacy library and "en_core_web_sm" database to run this file.
    nlp = spacy.load("en_core_web_sm", disable=['ner'])

    def tree_height(root):
        # get sentence max tree height
        if not list(root.children):
            return 1
        else:
            return 1 + max(tree_height(x) for x in root.children)

    def get_average_heights(paragraph):
        # get average tree height for post
        if type(paragraph) == str:
            doc = nlp(paragraph)
        else:
            doc = paragraph
        roots = [sent.root for sent in doc.sents]
        return np.mean([tree_height(root) for root in roots])

    return get_average_heights(post)

def max_verb_phrase(post):
    # NOTE: I referenced code at https://stackoverflow.com/questions/47856247/extract-verb-phrases-using-spacy
    # to create this function.
    nlp = spacy.load('en_core_web_sm')
    sentence = post
    pattern = [{'POS': 'VERB', 'OP': '?'},
           {'POS': 'ADV', 'OP': '*'},
           {'POS': 'AUX', 'OP': '*'},
           {'POS': 'VERB', 'OP': '+'}]

    # instantiate a Matcher instance
    matcher = Matcher(nlp.vocab)
    matcher.add("Verb phrase", [pattern])

    doc = nlp(sentence)
    # call the matcher to find matches
    matches = matcher(doc)
    spans = [doc[start:end] for _, start, end in matches]

    max = 0
    for i in filter_spans(spans):
        if str(i).count(" ") > max:
            max = str(i).count(" ")
    return max+1

# calculate first person pronoun counts
def fpp_count(post):
    cleaned_data = clean_data(post)
    # create a list of first person and other (secondary/tertiary) pronouns
    fpp = ["i", "me", "my", "mine", "we", "us", "our", "ours"]
    # calculate count of first person pronouns in the post
    first = 0
    for word in cleaned_data.split():
        if word in fpp:
            first += 1
    return first

def count_sentences(post):
    # count number of sentences using nltk library
    sentences = post
    number_of_sentences = sent_tokenize(sentences)
    return len(number_of_sentences)

def count_def_art(post):
    # count number of definite articles ("the" before a noun instances)
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(post)
    art_count = 0
    for np in doc.noun_chunks:
        lower_text = str(np).lower()
        if "the " in lower_text:
            art_count += 1
    return art_count

raw_data["AFINN_score"] = raw_data["Post"].apply(afinn_score)
raw_data["FPP_ratio"] = raw_data["Post"].apply(fpp_ratio_score)
raw_data["hrank_score"] = raw_data["Post"].apply(hrank_score)
raw_data["havg_score"] = raw_data["Post"].apply(havg_score)
raw_data["hstdv_score"] = raw_data["Post"].apply(hstdv_score)
raw_data["twit_score"] = raw_data["Post"].apply(twit_score)
raw_data["goog_score"] = raw_data["Post"].apply(goog_score)
raw_data["nyt_score"] = raw_data["Post"].apply(nyt_score)
raw_data["lyric_score"] = raw_data["Post"].apply(lyric_score)
raw_data["parse_tree_height"] = raw_data["Post"].apply(avg_tree_depth)
raw_data["verb_phrase_length"] = raw_data["Post"].apply(max_verb_phrase)
raw_data["FPP_count"] = raw_data["Post"].apply(fpp_count)
raw_data["count_sentence"] = raw_data["Post"].apply(count_sentences)
raw_data["count_def_articles"] = raw_data["Post"].apply(count_def_art)

final_data = raw_data[["User", "AFINN_score", "FPP_ratio", "hrank_score", "havg_score", "hstdv_score", "twit_score", "goog_score", "nyt_score", "lyric_score", "parse_tree_height", "verb_phrase_length", "FPP_count", "count_sentence", "count_def_articles"]]
final_data.to_csv("data/External_Features.csv", index=False)
