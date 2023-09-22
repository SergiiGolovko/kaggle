from __future__ import division
from jellyfish import damerau_levenshtein_distance

MIN_LEN = 0
DIST_THRESHOLD = 0.2


# def count_common_words(str1, str2, freq=False, len1=MIN_LEN):
#
#     words1, cnt = str1.split(), 0
#     for word1 in words1:
#         if len(word1) < len1:
#             continue
#
#         if word1 in str2:
#             if freq:
#                 cnt += str2.count(word1)
#             else:
#                 cnt += 1
#     return cnt



# consider making simpler but creating more features
def count_common_words(str1, str2, freq=False, len1=MIN_LEN):

    str1, str2 = str(str1), str(str2)
    words1, cnt = str1.split(), 0
    words2 = str2.split()
    for word1 in words1:
        if len(word1) < len1:
            continue

        if word1 in str2:
            if freq:
                cnt += str2.count(word1)
            else:
                cnt += 1
        else:
            found = False
            for word2 in words2:
                if (word2 in word1) and (len(word2) >= 4):
                    cnt += 1
                    found = True
                    # print word1, word2
                    if not freq:
                        break
            if not found:
                for word2 in words2:
                    distance = damerau_levenshtein_distance(unicode(word1), unicode(word2))
                    if distance / len(word1) <= DIST_THRESHOLD:
                        cnt += 1
                        #print word1, word2
                        if not freq:
                            break

    return cnt


# def length_common_words(str1, str2, len1=MIN_LEN):
#
#     words1, cnt = str1.split(), 0
#     for word1 in words1:
#         if len(word1) < len1:
#             continue
#
#         if word1 in str2:
#             cnt += len(word1)
#
#     return cnt


def length_common_words(str1, str2, len1=MIN_LEN):

    str1, str2 = str(str1), str(str2)
    words1, cnt = str1.split(), 0
    words2 = str2.split()
    for word1 in words1:
        if len(word1) < len1:
            continue

        if word1 in str2:
            cnt += len(word1)
        else:
            found = False
            for word2 in words2:
                if (word2 in word1) and (len(word2) >= 4):
                    cnt += len(word2)
                    found = True
                    # print word1, word2
                    break
            if not found:
                for word2 in words2:
                    distance = damerau_levenshtein_distance(unicode(word1), unicode(word2))
                    if distance / len(word1) <= DIST_THRESHOLD:
                        cnt += len(word1)
                        #print word1, word2
                        break
    return cnt


def count_whole_words(str1, str2, freq=False, len1=MIN_LEN):
    str1, str2 = str(str1), str(str2)
    words1, words2, cnt = str1.split(), str2.split(), 0
    for word in words1:
        if len(word) < len1:
            continue
        if word in words2:
            if freq:
                cnt += words2.count(word)
            else:
                cnt += 1
    return cnt


def length_whole_words(str1, str2, len1=MIN_LEN):
    str1, str2 = str(str1), str(str2)
    words1, words2, cnt = str1.split(), str2.split(), 0
    words1 = set(words1)
    words2 = set(words2)

    for word in words1:

        if len(word) < len1:
            continue

        if word in words2:
            cnt += len(word)
    return cnt


def count_words(s, l=MIN_LEN):
    s = str(s)
    words, cnt = s.split(), 0
    for word in words:
        if len(word) < l:
            continue
        # else
        cnt += 1
    return max(cnt, 1)


def length_words(s, l=MIN_LEN):
    s = str(s)
    words, cnt = s.split(), 0
    for word in words:
        if len(word) < l:
            continue
        # else
        cnt += len(word)
    return max(cnt, 1)
