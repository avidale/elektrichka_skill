import math
import os
import pandas as pd
import pickle
import requests
import tgalice

from datetime import datetime
from itertools import chain


import nltk
from nltk.stem import SnowballStemmer

STEMMER = SnowballStemmer(language='russian')
STOP_TOKENS = {'-', ',', '.', '(', ')', '№', '"', '«', '»'}


def tokenize(text, stem=False, join=False):
    text = text.lower()
    tokens = nltk.tokenize.wordpunct_tokenize(text)
    tokens = [t for t in tokens if t not in STOP_TOKENS]
    if stem:
        tokens = [STEMMER.stem(t) for t in tokens]
    if join:
        tokens = ' '.join(tokens)
    return tokens


class RaspDialogManager(tgalice.dialog_manager.base.BaseDialogManager):
    def __init__(self, world_filename, **kwargs):
        super(RaspDialogManager, self).__init__(**kwargs)
        with open(world_filename, 'rb') as f:
            world = pickle.load(f)
        self.regions = world['regions']
        self.settlements = world['settlements']
        self.stations = world['stations']

        self.searcher = RaspSearcher()

    def respond(self, ctx: tgalice.dialog.Context):
        response = tgalice.dialog.Response(
            text='Привет! Назовите две станции через пробел, например "Москва Петушки", '
                 'и я назову ближайшую электричку от первой до второй.'
        )
        if not ctx.message_text:
            return response

        toks = tokenize(ctx.message_text)
        if len(toks) != 2:
            return response

        name_from, name_to = toks

        search_results = self.find_from_to(name_from, name_to)

        if search_results is None:
            response.text = 'К сожалению, у меня не вышло найти электричек от {} до {}.'.format(name_from, name_to)
        else:
            response.text = phrase_results(search_results, name_from, name_to)
        return response

    def find_from_to(self, from_name, to_name):
        from_st, from_se = self.find_station(from_name)
        if from_st.shape[0] == 0 and from_se.shape[0] == 0:
            print('no from')
            return None
        to_st, to_se = self.find_station(to_name)
        if to_st.shape[0] == 0 and to_se.shape[0] == 0:
            print('no to')
            return None

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

        for i, pair in enumerate(candidate_pairs):
            results = self.searcher.suburban_trains_between(code_from=pair[0].yandex_code, code_to=pair[1].yandex_code)
            if len(results['segments']) > 0:
                print('success at {}!'.format(i))
                return results

    def find_station(self, station_name):
        stems = tokenize(station_name, stem=True, join=True)
        found_stations = self.stations[self.stations.stems == stems].copy()
        found_settlemens = self.settlements[self.settlements.stems == stems].copy()
        return found_stations, found_settlemens


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


def human_readable_time(time_string):
    ts = pd.to_datetime(time_string[:19])  # the last 6 symbols are timezone; we ignore them for now
    return '{}:{:02d}'.format(ts.hour, ts.minute)


class RaspSearcher:
    def __init__(self, token=None):
        self._token = token or os.getenv('RASP_TOKEN')
        self._cache = {}

    def suburban_trains_between(self, code_from, code_to, date=None):
        # see https://yandex.ru/dev/rasp/doc/reference/schedule-point-point-docpage/
        if date is None:
            date = str(datetime.now())[:10]  # todo: calculate 'now' in requester's timezone
        params = {
            'from': code_from,
            'to': code_to,
            'date': date,
            'transport_types': 'suburban',
        }
        key = str(sorted(params.items()))
        if key in self._cache:
            return self._cache[key]
        params['apikey'] = self._token
        print('calling api')
        rasp = requests.get('https://api.rasp.yandex.net/v3.0/search/', params=params)
        # todo: work with pagination
        result = rasp.json()
        self._cache[key] = result
        return result


def phrase_results(results, name_from, name_to):
    results_to_read = [
        seg for seg in results['segments']
        if pd.to_datetime(seg['departure']).to_pydatetime() > pd.datetime.now()
    ]
    if len(results_to_read) <= 0:
        pre = 'Сегодня все электрички от {} до {} ушли. Но вот какие были: в'.format(name_from, name_to)
        results_to_read = results['segments']
    else:
        pre = 'Вот какие ближайшие электрички от {} до {} есть: в'.format(name_from, name_to)
    times = [human_readable_time(r['departure']) for r in results_to_read]
    if len(times) == 1:
        pre = pre + ' {}'.format(times[0])
    else:
        pre = pre + ','.join([' {}'.format(t) for t in times[:-1]]) + ' и в' + ' {}'.format(times[-1])
    pre = pre + '.'
    return pre
