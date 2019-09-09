import numpy as np
import os
import pandas as pd
import pickle
import requests

from nlu import tokenize


def extract_all_objects(country):
    settlements = []
    stations = []
    regions = []
    for region_id, region in enumerate(country['regions']):
        region_code = region['codes'].get('yandex_code')
        regions.append({'title': region['title'], 'yandex_code': region_code, 'region_id': region_id})
        for settlement_id, settlement in enumerate(region['settlements']):
            settlement_code = settlement['codes'].get('yandex_code')
            latitude = np.median([s['latitude'] for s in settlement['stations'] if s['latitude']]) or None
            longitude = np.median([s['longitude'] for s in settlement['stations'] if s['longitude']]) or None
            settlements.append({
                'title': settlement['title'],
                'yandex_code': settlement_code,
                'region_id': region_id,
                'settlement_id': settlement_id,
                'latitude': latitude,
                'longitude': longitude,
            })
            for station_id, station in enumerate(settlement['stations']):
                station['region_id'] = region_id
                station['settlement_id'] = settlement_id
                stations.append(station)

    regions = pd.DataFrame(regions)
    settlements = pd.DataFrame(settlements)
    stations = pd.DataFrame(stations)

    stations['yandex_code'] = stations.codes.apply(lambda x: x.get('yandex_code'))

    return regions, settlements, stations


def prepare_the_world(stations_json):
    russia = None
    for c in stations_json['countries']:
        if c['title'] == 'Россия':
            russia = c
            break
    assert russia is not None, 'The target country Россия was not found in the data.'

    regions, settlements, stations = extract_all_objects(russia)

    print('regions: {}, settlements: {}, stations: {}'.format(regions.shape, settlements.shape, stations.shape))

    regions['stems'] = regions['title'].apply(tokenize, stem=True, join=True)
    settlements['stems'] = settlements['title'].apply(tokenize, stem=True, join=True)
    stations['stems'] = stations['title'].apply(tokenize, stem=True, join=True)

    regions['lemmas'] = regions['title'].apply(tokenize, lemma=True, join=True)
    settlements['lemmas'] = settlements['title'].apply(tokenize, lemma=True, join=True)
    stations['lemmas'] = stations['title'].apply(tokenize, lemma=True, join=True)

    world = {
        'regions': regions,
        'settlements': settlements,
        'stations': stations
    }
    return world


if __name__ == '__main__':
    token = os.getenv('RASP_TOKEN')
    assert token is not None, 'Pleas set the environment variable RASP_TOKEN'
    all_stations = requests.get('https://api.rasp.yandex.net/v3.0/stations_list/?apikey={}'.format(token))
    stations_json = all_stations.json()

    world = prepare_the_world(stations_json=stations_json)

    with open('../data/world.pkl', 'wb') as f:
        pickle.dump(world, f)
