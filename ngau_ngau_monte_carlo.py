import random
import pandas as pd
import numpy as np
import itertools
from collections import Counter
from tqdm import tqdm
from scipy.stats import kurtosis, skew
from scipy.stats.mstats import mquantiles
import statistics
import multiprocessing as mp
import json


# creating the deck
def create_deck():
    suits = ['C', 'D', 'H', 'S']
    numericals = [str(x) for x in range(1,11)] + ['J','Q','K']
    return [s+n for s in suits for n in numericals]


# random draw 5 cards
def get_random_card(seed):
    random.seed(seed)
    return random.sample(deck, 5)


# takes a list of cards and return a list of numerical values
def get_numeric(cards):
    tot = [int(c[-1]) if c[-1] not in ['J','Q','K'] else 0 for c in cards]
    return sum(tot) % 10


"""
function to arrange the cards for game-enable
brute force solution
"""
# returns formation in the format of array [[(3 cards),(2 cards)]]
def arrange(cards):
    # look for formation of 0 
    result = []
    for ipos, i in enumerate(cards):
        for jpos, j in enumerate(cards):
            if ipos == jpos:
                continue
            n = cards[:]
            if ipos < jpos:
                n.pop(jpos)
                n.pop(ipos)
            if ipos > jpos:
                n.pop(ipos)
                n.pop(jpos)
            if get_numeric(n) == 0 or len(set([c[-1] for c in n])) == 1:
                result.append(n)
    if result:
        # to remove duplicates
        arrangement_list = set(tuple(k) for k in result)
        return [[arr,tuple(set(cards)-set(arr))] for arr in arrangement_list]
    else:
        return result


def best_arrangement(cards):
    form = arrange(cards)
    # game-enabled scenario
    best_arr = None
    # the below checks hand from smallest to greatest
    if form:
        digits = [get_numeric(f[-1]) for f in form]
        best_arr = f'ngau-{max(digits)}' if all([d != 0 for d in digits]) else 'ngau-ngau'
        # check for ngau-pair
        for f in form:
            top = f[-1] # tuple of two cards
            if top[0][-1] == top[1][-1]:
                best_arr = 'ngau-pair'
                break
    # not game-enabled scenario
    else:
        pass
    
    # all-big and all-small
    cards_numeric = [str(c[-1]) for c in cards]
    all_big = ['0','J','Q','K']
    all_small = ['1','2','3','4']
    if all([n in all_small for n in cards_numeric]):
        best_arr = 'all-small'
    if all([n in all_big for n in cards_numeric]):
        best_arr = 'all-big'
    if best_arr is None:
        best_arr = 'no-game'
    return best_arr


# return winner for a list of high cards
def compare_sorted_list(a, b):
    if a and b:
        if a[0] > b[0]:
            return 0
        if a[0] < b[0]:
            return 1
        if a[0] == b[0]:
            return compare_sorted_list(a[1:], b[1:])
    else:
        return None


# comparison under no-game tie scenario
# return winner_idx
def compare_five_cards(h1, h2):
    winner_idx = None

    # check for number of pairs in the 5 cards
    h1_n = sorted([card_dict[h[1:]] for h in h1], reverse=True)
    h1_pairs = sorted([k for k, v in Counter(h1_n).items() if v > 1], reverse=True)

    h2_n = sorted([card_dict[h[1:]] for h in h2], reverse=True)
    h2_pairs = sorted([k for k, v in Counter(h2_n).items() if v > 1], reverse=True)

    if len(h1_pairs) > len(h2_pairs):
        winner_idx = 0
    elif len(h1_pairs) < len(h2_pairs):
        winner_idx = 1
    elif h1_pairs: # when len(h1_pairs) == len(h2_pairs) != 0
        winner_idx = compare_sorted_list(h1_pairs, h2_pairs)
    else: # when len(h1_pairs) == len(h2_pairs) == 0
        winner_idx = compare_sorted_list(h1_n, h2_n)
    
    return winner_idx


def payout_handler(winner_idx, winner_hand):
    # no payout for tie
    if winner_idx is None:
        return (0,0)
    else:
        mult = win_payout[winner_hand]
        payout = (mult, mult * -1) if winner_idx==0 else (mult * -1, mult)
        return payout


# input of h1, h2 should be two lists
# winner_idx = 0 means h1 wins; h2 if winner_idx = 1
# return a tuple representing result of pnl
def comparison(h1, h2):
    h1_hand = best_arrangement(h1)
    h2_hand = best_arrangement(h2)
    h1_pt = win_hierarchy[h1_hand]
    h2_pt = win_hierarchy[h2_hand]

    if h1_pt > h2_pt:
        winner_idx, winner_hand = 0, h1_hand
    elif h1_pt < h2_pt:
        winner_idx, winner_hand = 1, h2_hand
    else:
        # tie cases when both players got "no-game"
        if h1_pt == 0:
            winner_idx, winner_hand = compare_five_cards(h1, h2), 'no-game'
        else:
            winner_idx, winner_hand = None, h1_hand
    return payout_handler(winner_idx, winner_hand)


def get_random_card_for_players(player_num):
    return random.sample(deck, (player_num+1)*5)


def initialize_game(player_num):
    pnl = {'banker': []}
    for n in range(player_num):
        pnl[f'player-{n}'] = []
    return pnl


def one_session(rounds, player_num):
    pnl = initialize_game(player_num)
    for _ in range(rounds):
        hands_dict = {}
        # get all cards at one time as there are correlations between hands (draw from same deck)
        return_cards = get_random_card_for_players(player_num)
        for idx, k in enumerate(pnl.keys()):
            hands_dict[k] = return_cards[idx*5:(idx+1)*5]
        # banker compare with each player
        banker_payoff = 0
        for k, v in hands_dict.items():
            if k != 'banker':
                payoff = comparison(hands_dict['banker'], v)
                banker_payoff += payoff[0]
                pnl[k].append(payoff[1])
        pnl['banker'].append(banker_payoff)
    return {k: sum(v) for k,v in pnl.items()}


def play_sessions(rounds, player_num, sessions):
    pnl_list = []
    for _ in tqdm(range(sessions), desc=f'r={rounds}, n={player_num}'):
        pnl = one_session(rounds, player_num)
        pnl_list.append(pnl)
    return pnl_list



def stat_of(df, player):
    data = df[df.player==player]['pnl']
    mean = statistics.mean(data)
    var =  statistics.variance(data)
    sd = statistics.stdev(data)
    skewness = skew(data)
    kurt = kurtosis(data)
    qts = list(mquantiles(data, prob=np.linspace(0, 1, num=11)))
    return {'mean': mean, 'var': var, 'sd': sd, 'skewness': skewness, 'kurt': kurt, 'quantiles': qts}


def session_stat(session_result):
    # make analysis df
    dflist = []
    list_of_players = session_result[0].keys()

    for y in session_result:
        for k, v in y.items():
            dflist.append({'pnl':v, 'player':k})
    session_df = pd.DataFrame(dflist)
    
    session_stat_dict = {}
    for p in list_of_players:
        session_stat_dict[p] = stat_of(session_df, p)
    return session_stat_dict


"""
Reference of parallelization using mp
https://www.machinelearningplus.com/python/parallel-processing-python/
"""

# search for statistical result for change in parameters
def grid_search_once(n, r, sessions=10):
    tmp_session_result = play_sessions(r, n, sessions)
    return {str((n,r)): session_stat(tmp_session_result)}


def parallel_grid_search(player_num, rounds, sessions=1000):
    # parallelization
    param_list = [(n,r) for n in player_num for r in rounds]
    pool = mp.Pool(mp.cpu_count())

    sessions_results = []
    sessions_results = pool.starmap_async(grid_search_once, [(pl[0], pl[1]) for pl in param_list]).get()

    pool.close()
    final_result = {}
    for d in sessions_results:
        final_result.update(d)
    return final_result


def main():
    global deck
    global win_hierarchy
    global win_payout
    global card_dict

    deck = create_deck()

    ## Comparison of two different hands
    # setting up comparison dict by assigning explicit score to according to hand value
    win_hierarchy = {'all-big': 13, 'all-small': 12, 'ngau-pair': 11, 'ngau-ngau': 10, 'no-game': 0} 
    for x in range(1,10):
        win_hierarchy[f'ngau-{x}'] = x

    win_payout = {'all-big': 5, 'all-small': 5, 'ngau-pair': 5, 'ngau-ngau': 4, 'ngau-9': 3, 'ngau-8': 2, 'ngau-7': 2, 'no-game': 1}
    for x in range(1,7):
        win_payout[f'ngau-{x}'] = 1

    card_dict = {'J':11, 'Q':12, 'K': 13}
    for x in range(1,11):
        card_dict[str(x)] = x

    # simulation of playing between 1 chosen banker and n players
    player_num = list(range(2,9))
    rounds = list(range(20, 110, 10))
    sessions = 10000

    # player_num must be 2 or above; rounds must be >= 1
    grid = parallel_grid_search(player_num=player_num, rounds=rounds, sessions=sessions)

    # write to file
    with open('./ngau-ngau/result.json', 'w+') as fp:
        json.dump(grid, fp)


if __name__ == "__main__":
    main()