from mpi4py import MPI
import pandas as pd
from collections import Counter, defaultdict
import time
import re
import numpy as np
import os
import sys
import ijson


data_paths = {"tiny": "data/tinyTwitter.json",
              "small": "data/smallTwitter.json",
              "big": "data/bigTwitter.json"}

twitter_path = "data/smallTwitter.json"
args = sys.argv
if len(args) > 1:
    data_type = args[1]
    if data_type not in data_paths.keys():
        print("Invalid argument, tinyTwitter will be processed as default.")
    else:
        twitter_path = data_paths[data_type]

sal_path = "data/sal.json"

gcc_list = [
    "1gsyd",
    "2gmel",
    "3gbri",
    "4gade",
    "5gper",
    "6ghob",
    "7gdar",
    "8acte",
    "9oter"
]

gcc_full_list = [
    "1gsyd (Greater Sydney)",
    "2gmel (Greater Melbourne)",
    "3gbri (Greater Brisbane)",
    "4gade (Greater Adelaide)",
    "5gper (Greater Perth)",
    "6ghob (Greater Hobart)",
    "7gdar (Greater Darwin)",
    "8acte (Greater Canberra)",
    "9oter (Greater Other Territories)"
]

state_dict = {
    '1': 'New South Wales',
    '2': 'Victoria',
    '3': 'Queensland',
    '4': 'South Australia',
    '5': 'Western Australia',
    '6': 'Tasmania',
    '7': 'Northern Territory',
    '8': 'Australian Capital Territory',
    '9': 'Other Territories'
}

rural_pattern = r'^.[r].*'

json_start = b'^ {2}{'
json_end = b'^ {2}}'

tweet_head = b'^    "_id"'

pad_start = b'[\n'
pad_end = b'  }\n]\n'

batch_limit = 1024 * 1024 * 1024

rank_list = ['#1', '#2', '#3', '#4', '#5', '#6', '#7', '#8', '#9', '#10']


def create_int_dict():
    return defaultdict(int)


def load_and_process_json(file_path, size, rank, sal_data):
    file_size = os.path.getsize(file_path)
    chunk_size = file_size // size

    start = rank * chunk_size
    end = start + chunk_size if rank != size - 1 else file_size

    stat = {
        'tweets_cnt': [0] * 9,
        'top_users': Counter(),
        'cities_users': defaultdict(create_int_dict)
    }

    with open(file_path, 'rb') as file:
        start, end = get_batch_start_end(file, start, end, rank, size)
        file.seek(start)
        batch_size = end - start

        if batch_size > batch_limit:
            num_batch = (batch_size // batch_limit) + 1

            for i in range(num_batch):
                tmp_start = start + i * batch_limit
                tmp_end = tmp_start + batch_limit if i != num_batch - 1 else end
                tmp_start, tmp_end = get_piece_start_end(file, tmp_start, tmp_end, i, num_batch - 1)

                file.seek(tmp_start)
                json_raw = pad_start + file.read(tmp_end - tmp_start) + pad_end
                items = ijson.items(json_raw, 'item')

                for tweet in items:
                    analyze_tweet(tweet, sal_data, stat)

        else:
            json_raw = pad_start + file.read(end - start) + pad_end
            items = ijson.items(json_raw, 'item')

            for tweet in items:
                analyze_tweet(tweet, sal_data, stat)

    return stat


def get_batch_start_end(file, start, end, rank, size):
    file.seek(start)
    if start == 0:
        line = file.readline()
        start += len(line)
    else:
        while True:
            line = file.readline()
            if re.match(json_start, line):
                next_line = file.readline()
                if re.match(tweet_head, next_line):
                    break
                else:
                    start += len(next_line)
            start += len(line)

    if rank < size - 1:
        file.seek(end)
        while True:
            line = file.readline()
            if re.match(json_end, line):
                next_line = file.readline()
                if re.match(json_start, next_line):
                    end += len(line) - 6
                    break
                else:
                    end += len(next_line)
            end += len(line)
    else:
        end -= 6
    return start, end


def get_piece_start_end(file, start, end, index, tail):
    if index != 0:
        file.seek(start)
        while True:
            line = file.readline()
            if re.match(json_start, line):
                next_line = file.readline()
                if re.match(tweet_head, next_line):
                    break
                else:
                    start += len(next_line)
            start += len(line)

    if index != tail:
        file.seek(end)
        while True:
            line = file.readline()
            if re.match(json_end, line):
                next_line = file.readline()
                if re.match(json_start, next_line):
                    end += len(line) - 6
                    break
                else:
                    end += len(next_line)
            end += len(line)

    return start, end


def load_sal_data(file):
    data = {}
    with open(file, 'r', encoding='utf-8') as file:
        suburbs_data = ijson.items(file, '')
        for suburbs in suburbs_data:
            for suburb, values in suburbs.items():
                data[suburb.lower()] = {
                    'ste': values['ste'],
                    'gcc': values['gcc'],
                    'sal': values['sal'],
                    'ste_name': state_dict[values['ste']]
                }
    return data


def analyze_tweet(tweet, sal_data, stat):
    author_id = tweet['data']['author_id']
    full_name = tweet['includes']['places'][0]['full_name'].split(', ')
    suburb = full_name[0].lower()
    state = full_name[1].lower() if len(full_name) > 1 else ''

    if suburb in sal_data.keys() and sal_data[suburb]['ste_name'].lower() == state:
        gcc = sal_data[suburb]['gcc']
        if not re.match(rural_pattern, gcc):
            stat['tweets_cnt'][int(gcc[0]) - 1] += 1
            stat['top_users'][author_id] += 1
            stat['cities_users'][author_id][gcc] += 1


def process():
    start_time = time.time()
    # MPI settings
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    sal_data = load_sal_data(sal_path)
    batch = load_and_process_json(twitter_path, size, rank, sal_data)
    comm.Barrier()
    raw_twitter_stats = comm.gather(batch, root=0)

    if rank == 0:
        twitter_stats = [raw_twitter_stat for raw_twitter_stat in raw_twitter_stats]
        return twitter_stats, start_time
    return None


def print_stats(results, start_time):
    stat = {
        'tweets_cnt': [0] * 9,
        'top_users': Counter(),
        'cities_users': defaultdict(create_int_dict)
    }

    for result in results:

        stat['tweets_cnt'] = [stat['tweets_cnt'][i] + result['tweets_cnt'][i] for i in range(9)]
        stat['top_users'] += result['top_users']

        for author_id, cities in result['cities_users'].items():
            for key, value in cities.items():
                if key in stat['cities_users'][author_id]:
                    stat['cities_users'][author_id][key] += value
                else:
                    stat['cities_users'][author_id][key] = value

    top_users = stat['top_users'].most_common(10)
    most_cities_users = dict(sorted(stat['cities_users'].items(),
                                    key=lambda x: (len(x[1]), stat['top_users'][x[0]]), reverse=True))

    print("\nTask 1: Count the number of different tweets made in the Greater Capital cities of Australia")
    gcc_cnt = {"Greater Capital City": gcc_full_list, "Number of Tweets Made": stat["tweets_cnt"]}
    df_gcc_cnt = pd.DataFrame(data=gcc_cnt)
    print(df_gcc_cnt.to_string(index=False))

    print("\nTask 2: Identify the Twitter accounts (users) that have made the most tweets")
    df_top_user = pd.DataFrame(top_users, columns=['Author Id', 'Number of Tweets Made'])
    df_top_user.insert(0, 'Rank', rank_list, True)
    print(df_top_user.to_string(index=False))

    print("\nTask 3: Identify the users that have tweeted from the most different Greater Capital cities")
    n_uniq_city = []
    for i in list(most_cities_users.values())[0:10]:
        n_uniq_city.append(len(i))

    df_single_city_cnt = pd.DataFrame(data=list(most_cities_users.values())[0:10])
    df_single_city_cnt = df_single_city_cnt.reindex(sorted(df_single_city_cnt.columns), axis=1)
    df_single_city_cnt['total_tw'] = df_single_city_cnt.sum(axis=1)
    df_single_city_cnt['n_uniq_city'] = n_uniq_city
    df_single_city_cnt['Author Id'] = list(most_cities_users.keys())[0:10]
    df_scc_output = pd.DataFrame(data={'Rank': rank_list, 'Author Id': list(most_cities_users.keys())[0:10]})

    output_str_list = []
    for index, row in df_single_city_cnt.iterrows():
        r = row.dropna().astype(np.int64).astype(str)
        output_str = f'{row.n_uniq_city}(#{r.total_tw} tweets - '
        keys = list(r.keys())[:-3]
        values = list(r.values)[:-3]
        for i in range(len(keys)):
            output_str = output_str + values[i] + keys[i][1:] + ', '
        output_str = output_str[:-2] + ')'
        output_str_list.append(output_str)

    df_scc_output['Number of Unique City Locations and #Tweets'] = output_str_list
    print(df_scc_output.to_string(index=False))

    print(f"\nExecution Time: {round(time.time() - start_time, 2)}s\n\n")


if __name__ == '__main__':
    raw = process()

    if raw is not None:
        print_stats(raw[0], raw[1])
