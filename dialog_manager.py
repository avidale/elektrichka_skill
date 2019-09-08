import os
import pandas as pd
import pytz
import requests
import tgalice

from datetime import datetime

from nlu import tokenize, RouteMatcher

UTC = pytz.UTC


class RaspDialogManager(tgalice.dialog_manager.base.BaseDialogManager):
    def __init__(self, world_filename, **kwargs):
        super(RaspDialogManager, self).__init__(**kwargs)
        self.route_matcher = RouteMatcher(world_filename=world_filename)
        self.searcher = RaspSearcher()

    def respond(self, ctx: tgalice.dialog.Context):
        response = tgalice.dialog.Response(
            text='Привет! Назовите две станции, например "от Москвы до Петушков"'
                 'и я назову ближайшую электричку от первой до второй.'
        )
        if not ctx.message_text:
            return response

        parses = self.route_matcher.get_candidate_parses(ctx.message_text)

        toks = tokenize(ctx.message_text)
        if len(toks) == 2:
            parses.append({'from': toks[0], 'to': toks[1]})

        if len(parses) == 0:
            response.set_text('У меня не получилось понять ваш запрос. '
                              'Пожалуйста, назовите две станции, например, "Москва - Петушки" '
                              'или "От Сергиева Посада до Москвы".')
            return response

        for parse in parses:
            print('working with parse {}'.format(parse))
            candidate_pairs = self.route_matcher.get_candidate_pairs(from_name=parse['from'], to_name=parse['to'])
            for i, pair in enumerate(candidate_pairs):
                print('search {} - {}'.format(pair[0].title, pair[1].title))
                results = self.searcher.suburban_trains_between(code_from=pair[0].yandex_code,
                                                                code_to=pair[1].yandex_code)
                if len(results['segments']) > 0:
                    print('success at {}!'.format(i))
                    response.set_text(phrase_results(results, parse['from'], parse['to']))
                    return response
        parse = parses[0]
        response.set_text('К сожалению, у меня не вышло найти электричек от "{}" до "{}".'.format(
            parse['from'], parse['to']
        ))
        return response


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
    now = UTC.localize(pd.datetime.now())
    print('now is {}'.format(now))
    results_to_read = [
        seg for seg in results['segments']
        if pd.to_datetime(seg['departure']).to_pydatetime() > now
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
