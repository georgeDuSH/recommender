def file_reader_movie_lens_rating(path='', sep='::'):
    """ Read in rating dataset from movie lens

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


def obj_mapping(ratings):
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
    user2ix, _, item2ix, _ = obj_mapping(ratings)

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



