import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("Task 3 and 4_Loan_Data.csv")
df.head()
df.info()
df.describe()
fico_score = df["fico_score"].values
default = df["default"].values
def build_score_table(fico_score, default):

    df_temp = pd.DataFrame({'fico':fico_score, 'default': default})
    grouped = df_temp.groupby('fico').agg(
        n = ('default', 'count'),
        k = ('default', 'sum')
    ).reset_index()
    grouped = grouped.sort_values('fico').reset_index(drop = True)
    return grouped

score_table = build_score_table(fico_score, default)
def bucket_log_likelihood(n,k):
 
    if n == 0:
        return 0
    p = k/n
    if p == 1 or p == 0:
        return 0
    return k*np.log(p) + (n-k)*np.log(1 - p)
def compute_prefix_table(score_table):
    """
    Precompute cumulative n and k so that stats for any bucket [i, j)
    can be retrieved in O(1) instead of re-summing every time.
    """

    n_arr = score_table['n'].values
    k_arr = score_table['k'].values

    prefix_n = np.concatenate(([0], np.cumsum(n_arr)))#joins two arrays - concatenate
    prefix_k = np.concatenate(([0], np.cumsum(k_arr)))
    return prefix_n, prefix_k

def bucket_stats(prefix_n, prefix_k, i, j):
    """ 
    retuen (n, k) for the bucket covering score_table rows [i,j)
    """
    n = prefix_n[j] - prefix_n[i]
    k = prefix_k[j] - prefix_k[i]
    return n, k
def optimal_buckets_loglikelihood(score_table, num_buckets):
    """
    Find bucket boundaries over unique FICO scores that maximize total
    log-likelihood, using DP.
    
    State: dp[b][i] = best total log-likelihood using b buckets to cover
                       the first i unique scores (indices 0..i-1).
    Transition: dp[b][i] = max over j < i of dp[b-1][j] + LL(bucket [j, i))
    """
    U = len(score_table)  # number of unique FICO scores
    prefix_n, prefix_k = compute_prefix_table(score_table)
    
    # Precompute LL for every possible bucket [i, j) once, reused across b
    # ll_cache[i][j] = log-likelihood of bucket spanning indices i..j-1
    ll_cache = np.full((U + 1, U + 1), -np.inf)
    for i in range(U):
        for j in range(i + 1, U + 1):
            n, k = bucket_stats(prefix_n, prefix_k, i, j)
            ll_cache[i][j] = bucket_log_likelihood(n, k)
    
    # dp[b][i]: max total LL using b buckets over first i scores
    dp = np.full((num_buckets + 1, U + 1), -np.inf)
    dp[0][0] = 0  # 0 buckets, 0 scores covered -> LL = 0 (base case)
    
    # split[b][i]: the boundary j that achieves dp[b][i], for backtracking
    split = np.zeros((num_buckets + 1, U + 1), dtype=int)
    
    for b in range(1, num_buckets + 1):
        for i in range(1, U + 1):
            # try every possible start point j for the b-th bucket: [j, i)
            for j in range(b - 1, i):  # need at least b-1 scores for prior buckets
                if dp[b - 1][j] == -np.inf:
                    continue
                candidate = dp[b - 1][j] + ll_cache[j][i]
                if candidate > dp[b][i]:
                    dp[b][i] = candidate
                    split[b][i] = j
    
    return dp, split, score_table
def exact_boundaries(dp, split, score_table, num_buckets):
    """
    Walk backward through the split table to recover the actual
    FICO score boundaries.
    """

    u = len(score_table)
    boundaries_idx = []
    i = u
    b = num_buckets

    while b>0:
        j = split[b][i]
        boundaries_idx.append(j)
        i = j
        b -=1

    boundaries_idx.append(0)
    boundaries_idx = sorted(set(boundaries_idx))

    #convert index boyundaries into actual FICO score ut points

    fico_values = score_table['fico'].values
    score_boundaries = [fico_values[idx] for idx in boundaries_idx[:-1]]
    score_boundaries.append(fico_values[-1] + 1) #upper bound, exclusive

    return score_boundaries

best_ll = None 
def assign_ratings(fico_score, boundaries):
    """
    Map a single FICO score to a rating using the boundaries.
    Rating 0 = best (highest FICO), rating (num_buckets-1) = worst.
    """
    num_buckets = len(boundaries) - 1
    for i in range (num_buckets):
        if boundaries[i] <= fico_score < boundaries[i + 1]:
            
            
         # Buckets are ordered low-to-high FICO; we want low rating = high FICO
         # so we reverse the index   
            return num_buckets - 1 - i
    return None #score out of range

def build_rating_map(boundaries):
    num_buckets = len(boundaries) - 1
    rating_map = []
    for i in range(num_buckets):
        rating_map.append({
            'rating': num_buckets - 1 -i,
            'fico_min': boundaries[i],
            'fico_max': boundaries[i+1] - 1
        })
    return pd.DataFrame(rating_map).sort_values('rating').reset_index(drop=True)
def fit_fico_buckets(fico_scores, defaults, num_buckets):
    """
    Full pipeline: raw data -> optimal boundaries -> rating map.
    """
    score_table = build_score_table(fico_scores, defaults)
    dp, split, score_table = optimal_buckets_loglikelihood(score_table, num_buckets)
    boundaries = exact_boundaries(dp, split, score_table, num_buckets)
    rating_map = build_rating_map(boundaries)
    best_total_ll= dp[num_buckets][len(score_table)]

    return rating_map, boundaries, best_total_ll

def apply_rating_map(fico_scores, boundaries):
    return np.array([assign_ratings(s, boundaries) for s in fico_scores])
def fit_fico_buckets(fico_scores, defaults, num_buckets):
    """
    Full pipeline: raw data -> optimal boundaries -> rating map.
    """
    score_table = build_score_table(fico_scores, defaults)
    dp, split, score_table = optimal_buckets_loglikelihood(score_table, num_buckets)
    boundaries = exact_boundaries(dp, split, score_table, num_buckets)
    rating_map = build_rating_map(boundaries)
    best_total_ll= dp[num_buckets][len(score_table)]

    return rating_map, boundaries, best_total_ll

def apply_rating_map(fico_scores, boundaries):
    return np.array([assign_ratings(s, boundaries) for s in fico_scores])
