from mpi4py import MPI
import pandas as pd
import ijson
from collections import Counter, defaultdict
import time
import re


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


def create_int_dict():
    return defaultdict(int)


def load_twitter_data(file):
    with open(file, 'r', encoding='utf-8') as file:
        items = ijson.items(file, 'item')
        data = []
        for item in items:
            tweet = item['data']
            location = item['includes']['places'][0]
            full_name = location['full_name'].split(', ')
            data.append({
                'author_id': tweet['author_id'],
                'suburb': full_name[0],
                'state': full_name[1] if len(full_name) > 1 else '',
            })
        return data


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


def analyze(twitter_data, sal_data):
    res = {
        'tweets_cnt': [0] * 9,
        'top_users': Counter(),
        'cities_users': defaultdict(create_int_dict)
    }
    for tweet in twitter_data:
        author_id = tweet['author_id']
        suburb = tweet['suburb'].lower()
        state = tweet['state'].lower()

        if suburb in sal_data.keys() and sal_data[suburb]['ste_name'].lower() == state:
            gcc = sal_data[suburb]['gcc']
            if not re.match(rural_pattern, gcc):
                res['tweets_cnt'][int(gcc[0]) - 1] += 1
                res['top_users'][author_id] += 1
                res['cities_users'][author_id][gcc] += 1
    return res


def split_data(data, size):
    avg_len = len(data) // size
    return [data[i * avg_len:(i + 1) * avg_len] for i in range(size)]


def main():
    # MPI settings
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Create a new communicator with processes grouped by the nearest node
    local_comm = comm.Split_type(MPI.COMM_TYPE_SHARED, key=rank)
    local_rank = local_comm.Get_rank()
    local_size = local_comm.Get_size()

    start_time = time.time()

    twitter_data = load_twitter_data('data/smallTwitter.json')
    sal_data = load_sal_data('data/sal.json')

    # Split data into chunks for each process
    if rank == 0:
        data_chunks = split_data(twitter_data, local_size)
    else:
        data_chunks = None

    data_chunk = local_comm.scatter(data_chunks, root=0)
    batch = analyze(data_chunk, sal_data)
    local_comm.Barrier()
    results = local_comm.gather(batch, root=0)

    if rank == 0:
        stat = {
            'tweets_cnt': [0] * 9,
            'top_users': Counter(),
            'cities_users': defaultdict(create_int_dict)
        }

        for result in results:
            stat['tweets_cnt'] = [stat['tweets_cnt'][i] + result['tweets_cnt'][i] for i in range(9)]
            stat['top_users'] += result['top_users']

            for author_id, cities in result['cities_users'].items():
                stat['cities_users'][author_id].update(cities)

        top_users = stat['top_users'].most_common(10)
        most_cities_users = dict(sorted(stat['cities_users'].items(),
                                        key=lambda x: (len(x[1]), stat['top_users'][x[0]]), reverse=True))

        print("Task 1: Count the number of different tweets made in the Greater Capital cities of Australia")
        gcc_cnt = {"Greater Capital City": gcc_full_list, "Number of Tweets Made": stat["tweets_cnt"]}
        df_gcc_cnt = pd.DataFrame(data=gcc_cnt)
        print(df_gcc_cnt.to_string(index=False))

        print("\nTask 2: Identify the Twitter accounts (users) that have made the most tweets")
        cnt = 1
        for author_id, count in top_users:
            print(f"#{cnt} {author_id}: {count} tweets")
            cnt += 1

        cnt = 1
        print("\nTask 3: Identify the users that have tweeted from the most different Greater Capital cities")
        for author_id, cities in most_cities_users.items():
            if cnt == 11:
                break
            print(f"#{cnt} {author_id}: {stat['top_users'][author_id]} {cities.items()}")
            cnt += 1

        end_time = time.time()
        print(f"\nExecution Time: {round((end_time - start_time), 2)}s")


if __name__ == '__main__':
    main()
