synoms_dict = {}
split_dict = {}
split_words = {}

def replace_split_words(s):

    words = s.split()
    new_words = []

    for word in words:
        if word in split_dict:
            new_words.append(split_dict[word])
        else:
            new_words.append(word)

    return " ".join(new_words)

def create_dicts(str1, str2):
    global synonyms_dict, split_dict, split_words
    words1, words2 = set(str1.split()), set(str2.split())
    words2 = sorted(words2, key=len, reverse=True)
    # synonyms_dict = dict()
    # split_dict = dict()

    for word1 in words1:
        if word1 in words2:
            continue
        if len(word1) < 4:
            continue
        # otherwise find most similar word
        for word2 in words2:
            if len(word2) < 4:
                break
            if word1 in word2:
                if len(word2) - len(word1) > 3:
                    # add to dictionary
                    #if word1 in split_dict:
                    #    split_dict[word1].append(word2)
                    #else:
                    #    split_dict[word1] = {word2}

                    split_word = ""
                    if word2.startswith(word1):
                        split_word = word1 + " " + word2[len(word1):]
                    if word2.endswith(word1):
                        split_word = word2[:len(word2) - len(word1)] + " " + word1

                    # add to dictionary
                    if (word2 not in split_dict) & (split_word != ""):
                        split_dict[word2] = split_word



                    print word1, word2
                # len(word1) - len(word2) <= 3
                else:
                    # add to synonyms dictionary
                    if word1 in synonyms_dict:
                        synonyms_dict[word1].add(word2)
                    else:
                        synonyms_dict[word1] = {word2}

                    print word1, word2

                break

def remove_stopwords(s):

    global stopwords
    words = set(s.split())
    words = words - stopwords
    if len(words) == 0:
        words = set({"_n_u_l_l_"})
    return " ".join(words)


def remove_stopwords_df(df, columns):

    # return df
    print "Removing stopwords, it may take some time"
    time_start = time.time()
    for c in columns:
        df[c] = df[c].map(lambda s: remove_stopwords(s))
    print "Stopwords are removed, time elapsed" + str(time.time() - time_start) + " sec."
    return df


def previous_search_keys(df_all):

    print "previous search keys"

    df_importance3 = df_all[df_all['relevance'] >= 2.99][['product_uid', 'search_term']]
    df_unimportance1 = df_all[df_all['relevance'] <= 1.01][['product_uid', 'search_term']]

    df_important = df_importance3.groupby('product_uid', as_index=False).agg(lambda s: " ".join(s))
    df_unimportant = df_unimportance1.groupby('product_uid', as_index=False).agg(lambda s: " ". join(s))

    df_important = df_important.rename(columns={'search_term': 'important_previous_search'})
    df_unimportant = df_unimportant.rename(columns={'search_term': 'unimportant_previous_search'})

    df_all = pd.merge(df_all, df_important, how='left', on='product_uid')
    df_all = pd.merge(df_all, df_unimportant, how='left', on='product_uid')

    index_imp = ((~df_all['important_previous_search'].isnull()) & (df_all['relevance'].isnull()))
    print "number of important_previous_search %d" % sum(index_imp)

    index_unimp = ((~df_all['unimportant_previous_search'].isnull()) & (df_all['relevance'].isnull()))
    print "number of unimportant_previous_search %d" % sum(index_unimp)

    #df_all[df_all['important_previous_search'].isnull()]['important_previous_search'] = "null"

    #df_all = DataFrame()

    df_all['query_in_important'] = "n/tm"
    df_all['query_in_unimportant'] = "n/tm"
    df_all.loc[index_imp, 'query_in_important'] = df_all.loc[index_imp, 'search_term'] + '/t' + df_all.loc[index_imp, 'important_previous_search']
    df_all.loc[index_unimp, 'query_in_unimportant'] = df_all.loc[index_unimp, 'search_term'] + '/t' + df_all.loc[index_unimp, 'unimportant_previous_search']


    df_all['in_important_previous_search'] = df_all['query_in_important'].map(lambda x: count_whole_words(x.split('/t')[0],
                                                                                             x.split('/t')[1]))
    df_all['in_unimportant_previous_search'] = df_all['query_in_unimportant'].map(lambda x: count_whole_words(x.split('/t')[0],
                                                                                             x.split('/t')[1]))


    print "end previous search keys"

    return df_all


def remove_brand(title, brand):

    words1 = set(title)
    words2 = set(brand)
    words = words1 - words2
    if len(words) == 0:
        words = set({"_n_u_l_l_"})

    return " ".join(words)


def remove_brand_from_title(df_all):

    df = df_all['title'] + "/t" + df_all['brand']
    df_all['title'] = df.map(lambda s: remove_brand(s.split('/t')[0], s.split('/t')[1]))

    return df_all
