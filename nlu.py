import math
import nltk
import pickle
import re
import tgalice

from itertools import chain
from nltk.stem import SnowballStemmer

from functools import lru_cache
from pymorphy2 import MorphAnalyzer


STEMMER = SnowballStemmer(language='russian')
LEMMER = tgalice.nlu.basic_nlu.word2lemma
STOP_TOKENS = {'-', ',', '.', '(', ')', '№', '"', '«', '»'}


nonletters = re.compile('[^а-яёa-z0-9]+')

morph = MorphAnalyzer()


@lru_cache(10000)
def lemmatize(word):
    parses = morph.parse(word)
    if not parses:
        return word
    return parses[0].normal_form.replace('ё', 'е')


def normalize_address_text(text):
    return re.sub(nonletters, ' ', text.lower()).strip().replace('ё', 'е')


def tokenize(text, stem=False, lemma=False, join=False):
    text = normalize_address_text(text)
    tokens = nltk.tokenize.wordpunct_tokenize(text)
    tokens = [t for t in tokens if t not in STOP_TOKENS]
    if stem:
        tokens = [STEMMER.stem(t) for t in tokens]
    elif lemma:
        tokens = [LEMMER(t) for t in tokens]
    if join:
        tokens = ' '.join(tokens)
    return tokens


class RouteMatcher:
    def __init__(self, world_filename):
        with open(world_filename, 'rb') as f:
            world = pickle.load(f)
        self.regions = world['regions']
        self.settlements = world['settlements']
        self.stations = world['stations']

        self.expressions = []
        self.init_nlu()

    def get_candidate_parses(self, text):
        normalized_text = text.lower().strip()
        normalized_text = re.sub('[\s-]+', ' ', normalized_text)
        results = []
        for e in self.expressions:
            m = re.match(e, normalized_text)
            if m:
                results.append(m.groupdict())
        return results

    def init_nlu(self):
        re_prefix = '^(алиса )?(?:расписание )?(?:электриче?к[ауи]? )?'
        re_suffix = '( сегодня)?$'
        expressions = [
            re_prefix + '(?:от|из|с) (?P<from>[а-я0-9 ]{3,}) (?:до|в|к|на) (?P<to>[а-я0-9 ]{3,})' + re_suffix,
            re_prefix + '(?:до|в|к|на) (?P<to>[а-я0-9 ]{3,}) (?:от|из|с) (?P<from>[а-я0-9 ]{3,})' + re_suffix,
            re_prefix + '(?P<from>[а-я0-9]{3,}) (?P<to>[а-я0-9]{3,})' + re_suffix,  # 1 + 1
            re_prefix + '(?P<from>[а-я0-9]{3,} [а-я0-9]{3,}) (?P<to>[а-я0-9]{3,})$',  # 2 + 1
            re_prefix + '(?P<from>[а-я0-9]{3,}) (?P<to>[а-я0-9]{3,} [а-я0-9]{3,})' + re_suffix,  # 1 + 2
            re_prefix + '(?P<from>[а-я0-9]{3,} [а-я0-9]{3,}) (?P<to>[а-я0-9]{3,} [а-я0-9]{3,})' + re_suffix,  # 2 + 2
        ]
        self.expressions = [re.compile(e) for e in expressions]

    def get_candidate_pairs(self, from_name, to_name):
        from_st, from_se = self.find_station(from_name)
        if from_st.shape[0] == 0 and from_se.shape[0] == 0:
            print('no from: ')
            return []
        to_st, to_se = self.find_station(to_name)
        if to_st.shape[0] == 0 and to_se.shape[0] == 0:
            print('no to')
            return []

        candidate_pairs = []
        for i, row1 in chain(from_st.iterrows(), from_se.iterrows()):
            if not row1.yandex_code:
                continue
            for j, row2 in chain(to_st.iterrows(), to_se.iterrows()):
                if not row2.yandex_code:
                    continue
                distance = geo_distance(row1.latitude, row1.longitude, row2.latitude, row2.longitude) or 9999
                candidate_pairs.append((row1, row2, distance))
        assert len(candidate_pairs) > 0

        candidate_pairs.sort(key=lambda x: x[2])

        return candidate_pairs

    def find_station(self, station_name):
        stems = tokenize(station_name, stem=True, join=True)
        lemmas = tokenize(station_name, lemma=True, join=True)
        found_stations = self.stations[(self.stations.stems == stems) | (self.stations.lemmas == lemmas)].copy()
        found_settlements = self.settlements[
            (self.settlements.stems == stems) | (self.settlements.lemmas == lemmas)
        ].copy()
        return found_stations, found_settlements


def geo_distance(lat1, lon1, lat2, lon2):
    # approximate radius of earth in km
    r = 6373.0
    try:
        lat1 = math.radians(lat1)
        lon1 = math.radians(lon1)
        lat2 = math.radians(lat2)
        lon2 = math.radians(lon2)
    except TypeError:
        return None
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = r * c
    return distance
