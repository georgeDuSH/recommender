import sys, os
sys.path.append(os.getcwd())

def file_reader_movie_lens_rating(path='', sep='::'):
    """ Read in rating datasets from movie lens

    :param path:
    :param sep:
    :return:
    """
    f = open(file=path, mode='r')
    if sep==',':
        f.readline() # skip header
    rats = []
    f_line = f.readline()
    while True:
        if not f_line:
            break
        # parse
        user, item, rating, timestamp = f_line.strip().split(sep)
        rats.append([user, item, rating])

        f_line = f.readline()

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

    users, items, _ = zip(*ratings)
    user2ix, ix2user = obj2ix(users)
    item2ix, ix2item = obj2ix(items)

    return user2ix, ix2user, item2ix, ix2item

def rating_train_test_parser(ratings, perc=0.2, test_filter=True):
    """ parse rating into training and testing set

    :param ratings: list
        A list of triplet denoting users' rating records on movies

    :param perc: float
        The percentage that should is divided into testing set

    :param test_filter: bool
        If we remove the rating record from testing set if the rating is smaller than 3

    :return rat_train_dict, rat_test_dict: dict
         A dictionary of users rating on items.
         {userix: {movieix: rating, ...}, ...}
    """

    sep_counter = int(1/perc)
    rat_train_dict = dict()
    rat_test_dict = dict()
    user2ix, _, item2ix, _ = mapping(ratings)

    for ix, rat in enumerate(ratings):
        u, i, r = rat
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

    users_of_interest = list(rat_test_dict.keys())
    ratings_of_interest = [rat_train_dict[user] for user in users_of_interest]
    rat_train_dict = dict(zip(users_of_interest, ratings_of_interest))

    return rat_train_dict, rat_test_dict

def load_ml_rating(path, sep, need_raw, need_split):
    """ Load rating from movie lens datasets

    :param path: str, path like
        relative path of rating file

    :param sep: str
        separation comma of the file

    :return: dict
    """
    ratings = file_reader_movie_lens_rating(path=path, sep=sep)
    user2ix, ix2user, item2ix, ix2item = mapping(ratings)
    n_user = len(user2ix.keys())
    n_item = len(item2ix.keys())
    rat_train_dict, rat_test_dict = rating_train_test_parser(ratings)

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
def load_ml_small_rating(path='./recom/datasets/ml-small/ratings.csv', need_raw=True, need_split=True):
    return load_ml_rating(path=path, sep=',', need_raw=need_raw, need_split=need_split)

def load_ml_1m_rating(path='./recom/datasets/ml-1m/ratings.dat', need_raw=True, need_split=True):
    return load_ml_rating(path=path, sep='::', need_raw=need_raw, need_split=need_split)

#
# if __name__=='__main__':
#     print(os.getcwd())
#     data = load_ml_small_rating()
#     print(len(data['ix2item']))
#     print(len(data['ix2user']))
#     data = load_ml_1m_rating()
#     print(len(data['ix2item']))
#     print(len(data['ix2user']))
