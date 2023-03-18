from mpi4py import MPI
import pandas as pd
import ijson
from collections import Counter, defaultdict
import time


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

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


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
        data = pd.DataFrame(data)
        data = data.dropna(subset=['author_id'])
        return data


def load_sal_data(file):
    data = []
    with open(file, 'r', encoding='utf-8') as file:
        suburbs_data = ijson.items(file, '')
        for suburbs in suburbs_data:
            for suburb, values in suburbs.items():
                data.append({
                    'suburb': suburb,
                    'ste': values['ste'],
                    'gcc': values['gcc'],
                    'sal': values['sal'],
                    'ste_name': state_dict[values['ste']]
                })
    return pd.DataFrame(data)


def analyze(twitter_data, sal_data):
    res = {
        'tweets_count': 0,
        'top_users': Counter(),
        'cities_users': defaultdict(set)
    }
    for _, tweet in twitter_data.iterrows():
        author_id = tweet['author_id']
        suburb = tweet['suburb']
        state = tweet['state']
        matched_suburb = sal_data[(sal_data['suburb'].str.lower() == suburb.lower()) &
                                  (sal_data['ste_name'].str.lower() == state.lower())]
        if not matched_suburb.empty:
            gcc = matched_suburb['gcc'].iloc[0]
            # Ignore rural areas
            if not gcc.startswith("1r"):
                res['tweets_count'] += 1
                res['top_users'][author_id] += 1
                res['cities_users'][author_id].add(gcc)
    return res


def parallel_processing(data):
    chunk_size = len(data) // size
    start = rank * chunk_size
    if rank != size - 1:
        end = (rank + 1) * chunk_size
    else:
        end = len(data)
    return data[start:end]


def main():
    twitter_data = load_twitter_data('data/twitter-data-small.json')
    sal_data = load_sal_data('data/sal.json')

    twitter_data_chunk = parallel_processing(twitter_data)

    batch = analyze(twitter_data_chunk, sal_data)
    results = comm.gather(batch, root=0)

    if rank == 0:
        stat = {
            'tweets_count': 0,
            'top_users': Counter(),
            'cities_users': defaultdict(set)
        }

        for result in results:
            stat['tweets_count'] += result['tweets_count']
            stat['top_users'] += result['top_users']

            for author_id, cities in result['cities_users'].items():
                stat['cities_users'][author_id].update(cities)

        top_users = stat['top_users'].most_common(10)
        most_cities_users = sorted(stat['cities_users'].items(), key=lambda x: len(x[1]), reverse=True)[:10]

        print("Number of tweets in Greater Capital cities:", stat['tweets_count'])

        print("\nTop 10 tweeters")
        cnt = 1
        for author_id, count in top_users:
            print(f"#{cnt} {author_id}: {count} tweets")
            cnt += 1

        cnt = 1
        print("\nTop 10 tweeters making tweets from the most different locations")
        for author_id, cities in most_cities_users:
            print(f"#{cnt} {author_id}: {cities}")
            cnt += 1


if __name__ == '__main__':
    main()
