"""
__file__

    ngram.py

__description__

    This file provides functions to compute n-gram & n-term.

__author__

    Corrected from Chenglong Chen < c.chenglong@gmail.com > original file

"""

def getBigram(s, join_string="_", skip=0):
    """
	   Input: a list of words, e.g., ['I', 'am', 'Denny']
	   Output: a list of bigram, e.g., ['I_am', 'am_Denny']
	   I use _ as join_string for this example.
	"""
#    assert type(s) == str
    s = str(s)
    words = s.split()
    L = len(words)
    if L > 1:
        lst = []
        for i in range(L-1):
            for k in range(1, skip+2):
                if i+k < L:
                    lst.append( join_string.join([words[i], words[i+k]]) )
    else:
        # set it as unigram
		lst = words
    return " ".join(word for word in lst)
