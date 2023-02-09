import sys, os

sys.path.append(os.getcwd())

def file_reader_movie_lens_rating(path='', sep='::', time_ord=False):
    """ Read in rating datasets from movie-lens.
        Set time_ord to True, will make further splits by time order.

    :param path:
    :param sep:
    :return:
    """
    f = open(file=path, mode='r')
    if sep==',':
        f.readline() # skip header
    rats = []
    f_line = f.readline()

    if not time_ord:
        while True:
            if not f_line:
                break
            # parse
            user, item, rating, timestamp = f_line.strip().split(sep)
            rats.append([user, item, rating, timestamp])

            f_line = f.readline()
    
    else:
        last_user_hist = []
        while True:
            if not f_line:
                break
            # parse
            user, item, rating, timestamp = f_line.strip().split(sep)
            try: # last_user_hist is not null
                if last_user_hist[-1][0]==user:
                    last_user_hist.append([user, item, rating, timestamp])
                else:
                    sorted_rat = sorted(last_user_hist, key=lambda x: x[-1])
                    rats.extend([record for record in sorted_rat])
                    last_user_hist = []

            except: # fail to index, last_user_hist is null
                last_user_hist.append([user, item, rating, timestamp])

            f_line = f.readline()

        last_user_hist.append([user, item, rating, timestamp]) # add last line
        sorted_rat = sorted(last_user_hist, key=lambda x: x[-1])
        rats.extend([record for record in sorted_rat])
        
    return rats


def mapping(ratings):
    """ Read in rating and transform into map

    :param rats:
    :return:
    """
    def obj2ix(obj):
        obj = sorted(list(set(obj)), key=lambda x: int(x))
        ix = list(range(len(obj)))
        return dict(zip(obj, ix)), dict(zip(ix, obj))

    users, items, _, __ = zip(*ratings)
    user2ix, ix2user = obj2ix(users)
    item2ix, ix2item = obj2ix(items)

    return user2ix, ix2user, item2ix, ix2item


def rating_train_test_parser(ratings, perc=0.2, test_filter=True, time_ord=False):
    """ parse rating into training and testing set

    :param ratings: list
        A list of triplet denoting users' rating records on movies

    :param perc: 'ooc' or float
        The percentage that should is divided into testing set

    :param test_filter: bool
        If we remove the rating record from testing set if the rating is smaller than 3

    :return rat_train_dict, rat_test_dict: dict
         A dictionary of users rating on items.
         {userix: {movieix: rating, ...}, ...}
    """

    rat_train_dict = dict()
    rat_test_dict = dict()
    user2ix, _, item2ix, _ = mapping(ratings)

    # split train test randomly
    if not time_ord: 
        sep_counter = int(1/perc)

        for ix, rat in enumerate(ratings):
            u, i, r, _ = rat
            uix = user2ix[u]
            iix = item2ix[i]
            r = float(r)
            if ix % sep_counter == 0:
                # skip the record if its rating is lower than 3
                if test_filter and r<3:
                    continue
                if uix not in rat_test_dict:
                    rat_test_dict[uix] = {}
                rat_test_dict[uix][iix] = r
            else:
                if uix not in rat_train_dict:
                    rat_train_dict[uix] = {}
                rat_train_dict[uix][iix] = r

    # split train test by time
    else:
        for ix, rat in enumerate(ratings):
            u, i, r, _ = rat
            uix = user2ix[u]
            iix = item2ix[i]
            r = float(r)
            
            if r<3 and test_filter:
                continue

            if uix not in rat_train_dict:
                rat_train_dict[uix] = {}
            rat_train_dict[uix][iix] = r
            
        for uix in rat_train_dict:
            rat_count = len(rat_train_dict[uix])
            split_point = rat_count-(1 if perc=='loo' else int(rat_count*perc))
            rat_train_dict[uix], rat_test_dict[uix] = dict(list(rat_train_dict[uix].items())[:split_point]) \
                                                      , dict(list(rat_train_dict[uix].items())[split_point:])

    users_of_interest = list(rat_test_dict.keys())
    ratings_of_interest = [rat_train_dict[user] for user in users_of_interest]
    rat_train_dict = dict(zip(users_of_interest, ratings_of_interest))

    return rat_train_dict, rat_test_dict


def load_ml_rating(path, sep
                   , time_ord, test_perc
                   , need_raw, need_split, test_filter):
    """ Load rating from movie lens datasets

    :param path: str, path like
        relative path of rating file

    :param sep: str
        separation comma of the file

    :return: dict
    """
    ratings = file_reader_movie_lens_rating(path=path, sep=sep, time_ord=time_ord)
    user2ix, ix2user, item2ix, ix2item = mapping(ratings)
    n_user = len(user2ix.keys())
    n_item = len(item2ix.keys())
    rat_train_dict, rat_test_dict = rating_train_test_parser(ratings, perc=test_perc, time_ord=time_ord, test_filter=test_filter)

    dataset = {
        'n_user': n_user
        , 'n_item': n_item
        , 'user2ix': user2ix
        , 'ix2user': ix2user
        , 'item2ix': item2ix
        , 'ix2item': ix2item
    }

    if need_raw:
        dataset['raw'] = ratings

    if need_split:
        dataset['train_dict'] = rat_train_dict
        dataset['test_dict'] = rat_test_dict

    return dataset


# suppose we load from root
def load_ml_small_rating(path='./recom/datasets/ml-small/ratings.csv'
                         , time_ord=False, test_perc=0.2
                         , need_raw=True, need_split=True, test_filter=True):

    return load_ml_rating(path=path, sep=','
                          , time_ord=time_ord, test_perc=test_perc
                          , need_raw=need_raw, need_split=need_split, test_filter=test_filter)


def load_ml_1m_rating(path='./recom/datasets/ml-1m/ratings.dat'
                      , time_ord=False, test_perc=0.2
                      , need_raw=True, need_split=True, test_filter=True):

    return load_ml_rating(path=path, sep='::'
                          , time_ord=time_ord, test_perc=test_perc
                          , need_raw=need_raw, need_split=need_split, test_filter=test_filter)


def load_movie_genres(path, sep=','):

    f = open(file=path, mode='r', encoding='utf-8')

    if sep==',':
        f.readline() # skip header
    genres = {'g_base':dict(), 'g_set':set(), 'g2ix':dict(), 'ix2g':dict(), 'g_vec':dict()}
    f_line = f.readline()

    while True:
        if not f_line:
            break
        # parse
        raw = f_line.strip().split(sep)
        # add genre to dict
        genres['g_base'][raw[0]] = raw[-1]
        # add to set
        for g in raw[-1].split('|'):
            genres['g_set'].add(g)
        f_line = f.readline()

    # vectorize
    genres['g_set'] = sorted(list(genres['g_set']))
    genres['g2ix'] = dict(zip(genres['g_set'], range(len(genres['g_set']))))
    genres['ix2g'] = dict(zip(range(len(genres['g_set'])), genres['g_set']))
    
    zero_vec = [0 for _ in range(len(genres['g_set']))]
    
    for movie in genres['g_base']:
        genre_vec = zero_vec[:]
        for g in genres['g_base'][movie].split('|'):
            if g != '(no genres listed)':
                genre_vec[genres['g2ix'][g]] = 1
        # overwrite
        genres['g_vec'][movie] = genre_vec

    return genres


#
# if __name__=='__main__':
#     print(os.getcwd())
#     data = load_ml_small_rating()
#     print(len(data['ix2item']))
#     print(len(data['ix2user']))
#     data = load_ml_1m_rating()
#     print(len(data['ix2item']))
#     print(len(data['ix2user']))
