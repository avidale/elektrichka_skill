import pandas as pd
import pytz
import tgalice

from grammars import calculate_spans
from nlu import tokenize, RouteMatcher, lemmatize
from rasp_api import RaspSearcher

UTC = pytz.UTC

import time


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

        words = ctx.message_text.lower().split()
        lemmas = [lemmatize(w) for w in words]

        slots = calculate_spans(lemmas, return_span=True, parser=self.parser)
        if slots is None:
            # если грамматикой попарсить запрос не получилось, запускаем более медленную и универсальную нейронку.
            print('applying the neural network...')
            slots = predict_slots(words)
        else:
            # от лемм возвращаемся к исходным словам, так чуть проще искать будет, т.к. обработка текстов сделается Searcher'ом в его стиле.
            slots = {k: [' '.join(words[span[0]: span[1]]) for span in v] for k, v in slots.items()}
        print(slots)

        response_text = 'Если бот вернул вам этот текст, то я забыл дописать какой-то if.'
        if 'FROM_PLACE' in slots and 'TO_PLACE' in slots:
            from_text = slots['FROM_PLACE'][0]
            to_text = slots['TO_PLACE'][0]
            maybe_from = match_geo(from_text)
            maybe_to = match_geo(to_text)
            if maybe_from.shape[0] == 0:
                response = 'Не могу понять, что это за станция - "{}".'.format(from_text)
            elif maybe_to.shape[0] == 0:
                response = 'Не могу понять, что это за станция - "{}".'.format(to_text)
            else:
                results = self.searcher.suburban_trains_between(
                    maybe_from.yandex_code.iloc[0],
                    maybe_to.yandex_code.iloc[0]
                )
                response_text = phrase_results(results, maybe_from.title.iloc[0], maybe_to.title.iloc[0])
                url = 'https://rasp.yandex.ru/search/suburban/?fromId={}&toId={}'.format(maybe_from.yandex_code.iloc[0],
                                                                                         maybe_to.yandex_code.iloc[0])
                response_text += '\n <a href="{}">Маршрут на Яндекс.Расписаниях</a>'.format(url)
        else:
            response_text = 'У меня не получилось разобрать ваш запрос. ' \
                            'Пожалуйста, назовите станции отправления и назначения.'
        response.set_text(response_text)
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


def phrase_results(results, name_from, name_to, max_results=10):
    """ Превращаем ответ API электричек в связный текст. """
    now = pytz.UTC.localize(pd.datetime.now())
    # print('now is {}'.format(now))
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
    if len(results_to_read) == 0:
        return 'Такого маршрута у меня не нашлось.'

    if len(times) == 1:
        pre = pre + ' {}'.format(times[0])
    else:
        if len(times) < max_results:
            last_id = -1
        else:
            last_id = max_results - 1
        pre = pre + ','.join([' {}'.format(t) for t in times[:max_results]]) + ' и в' + ' {}'.format(times[-1])
    pre = pre + '.'
    if len(times) > max_results:
        pre = pre + ' И ещё другие; всего {} вариантов, но я столько за раз не прочитаю.'.format(len(times))
    return pre

