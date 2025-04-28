
# Nom(s) étudiant(s) / Name(s) of student(s):
# Yorguin José Mantilla Ramos
# Matricule: 20253616

import sys

# Espace pour fonctions auxillaires :
# Space for auxilary functions :

def insertion_sort(A, start, end):
    """Based on:
    Cormen, Thomas H., author. Introduction to algorithms / Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, Clifford Stein. Fourth edition. Cambridge, Massachusett : The MIT Press, [2022]

    Modified so that it sorts the array A from index start to end (inclusive), and that it does it in place.
    """
    # Modifies in-place the array A from index start to end (inclusive) using insertion sort.
    for i in range(start + 1, end + 1):
        key = A[i]
        # Insert A[i] into sorted subarray A[start:i-1]
        j = i - 1
        while j >= start and A[j] > key:
            A[j + 1] = A[j]
            j -= 1
        A[j + 1] = key

def merge(A, d, m, f):
    """Based on class notes and slides (IFT2125).

    Updated some conditions as python is 0-indexed. Does it in place.
    """
    # Auxiliary arrays initialization
    nleft = m - d + 1
    nRight = f - m
    left = [0] * nleft
    right = [0] * nRight
    # Copy data to auxiliary arrays
    for i in range(0, nleft):
        left[i] = A[d + i]
    for j in range(0, nRight):
        right[j] = A[m + 1 + j]
    
    il = 0 # index left
    ir = 0 # index right
    im = d # index merged

    while il < nleft and ir < nRight:
        if left[il] <= right[ir]:
            A[im] = left[il]
            il += 1
        elif left[il] > right[ir]:
            A[im] = right[ir]
            ir += 1
        im += 1

    # Handle leftover elements

    while il < nleft:
        A[im] = left[il]
        il += 1
        im += 1
    while ir < nRight:
        A[im] = right[ir]
        ir += 1
        im += 1

def hybrid_sort(A, d, f, threshold=24):
    """Based on class notes and slides (IFT2125).

    Updated some conditions as python is 0-indexed. Does it in place.
    Note that if threshold is -1, it will be a pure merge sort.
    """
    # Sorts the array A from index d to f (inclusive) using hybrid sort.

    if d >= f:
        return
    
    if f - d <= threshold:
        if DEBUG:
            print('Insertion sort from', d, 'to', f)
        insertion_sort(A, d, f)
    else:
        if DEBUG:
            print('Merge sort from', d, 'to', f)
        m = (d + f) // 2
        hybrid_sort(A, d, m, threshold)
        hybrid_sort(A, m + 1, f, threshold)
        merge(A, d, m, f)

### This part of the code is for my own testing and debugging purposes.
### also includes the code to determine the best threshold

DEBUG = False
if DEBUG:

    ## We will use PI digits to test the sorting algorithms.
    from decimal import Decimal, getcontext

    # Compute pi using the Chudnovsky algorithm
    def compute_pi(digits):
        # Increase precision a bit to avoid rounding issues
        getcontext().prec = digits + 10
        
        C = 426880 * Decimal(10005).sqrt()
        M = 1
        L = 13591409
        X = 1
        K = 6
        S = L

        for i in range(1, digits):
            M = (M * (K**3 - 16*K)) // (i**3)
            L += 545140134
            X *= -262537412640768000
            S += Decimal(M * L) / X
            K += 12

        pi = C / S
        return pi

    def get_pi(n=15):
        pi = compute_pi(n)
        pi_str = str(pi)
        # Remove the decimal point and take the first n digits
        digits = [int(c) for c in pi_str if c.isdigit()][:n]
        return digits


    ## Quick test to see if the sorting algorithms work correctly.
    pi_list = get_pi()
    print("pi_list avant tri : ", get_pi())
    insertion_sort(pi_list, 0, len(pi_list) - 1)
    assert pi_list == sorted(pi_list)
    print("pi_list après tri : ", pi_list)


    pi_list = get_pi()
    print("pi_list avant tri : ", get_pi())
    hybrid_sort(pi_list, 0, len(pi_list) - 1, 3)
    assert pi_list == sorted(pi_list)
    print("pi_list après tri : ", pi_list)

    # determining the best threshold
    import time

    MAX_ORDER = 3
    N_TESTS = 10

    range_digits = []

    for i in range(1, MAX_ORDER + 1):
        base = 10 ** i
        half = 5 * (10 ** (i-1))  # half between 10^i and 10^(i+1)
        range_digits.append(10 ** i)
        range_digits.append(half)

    range_digits = sorted(range_digits)
    print("range_digits : ", range_digits)

    def do_tests(range_digits, N_TESTS, thr=None):
        insertion_times = {}
        mergesort_times = {}

        for i in range_digits:
            insertion_times[i] = []
            mergesort_times[i] = []
            for j in range(N_TESTS):
                print(i,j, flush=True,end=" ")

                if thr is None: # if no threshold is given, we do not insertion (is inside hybrid_sort)
                    # Get time for insertion sort
                    print("insertion sort, no threshold given")
                    pi_list = get_pi(i)
                    len_pi_list = len(pi_list)
                    start = time.time()
                    insertion_sort(pi_list, 0, len_pi_list - 1)
                    end = time.time()
                    insertion_sort_time = end - start
                    insertion_times[i].append(insertion_sort_time)

                if thr is None:
                    print("pure merge sort")
                    real_thr = -1 # pure merge sort
                else:
                    print("hybrid sort with threshold", thr)
                    real_thr = thr
                # Get time for merge sort
                pi_list = get_pi(i)
                len_pi_list = len(pi_list)
                start = time.time()
                hybrid_sort(pi_list, 0, len_pi_list - 1, real_thr)
                end = time.time()
                mergesort_time = end - start
                mergesort_times[i].append(mergesort_time)

        if thr is None:
            return insertion_times, mergesort_times
        else:
            return mergesort_times

    # Plotting the results
    # use boxplots to show the distribution of the times per order of magnitude
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import pandas as pd
    import os
    if not os.path.exists("sorting_time_comparison.png"):
        insertion_times, mergesort_times = do_tests(range_digits, N_TESTS)

        df_insertion = pd.DataFrame(insertion_times).T
        df_mergesort = pd.DataFrame(mergesort_times).T
        df_insertion.to_csv("insertion_times.csv")
        df_mergesort.to_csv("mergesort_times.csv")
    else:
        print("Loading previous results...")
        df_insertion = pd.read_csv("insertion_times.csv", index_col=0)
        df_mergesort = pd.read_csv("mergesort_times.csv", index_col=0)

    def plot(df_insertion,df_mergesort,filename="sorting_time_comparison.png", log=False,figsize=(12, 6)):
        range_digits = df_insertion.index.astype(int).tolist()

        # Prepare data: mean and std dev for each order of magnitude
        mean_insertion = df_insertion.mean(axis=1)
        std_insertion = df_insertion.std(axis=1)

        mean_mergesort = df_mergesort.mean(axis=1)
        std_mergesort = df_mergesort.std(axis=1)

        # Plot
        fig, ax = plt.subplots(figsize=figsize)

        # Plot Insertion sort
        ax.errorbar(
            range_digits, 
            mean_insertion, 
            yerr=std_insertion, 
            fmt='o-', 
            label='Insertion Sort', 
            capsize=5
        )

        # Plot Merge sort
        ax.errorbar(
            range_digits, 
            mean_mergesort, 
            yerr=std_mergesort, 
            fmt='s--', 
            label='Merge Sort', 
            capsize=5
        )


        if log:
            ax.set_xscale('log')

        ax.set_xticks(range_digits)
        ax.set_xticklabels(range_digits, rotation=90)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())  # Show ticks as 10, 50, 100 instead of 10^1
        ax.tick_params(axis='x', which='both', length=5)

        ax.set_xlabel("Order of Magnitude (Input Size)")
        ax.set_ylabel("Time (s)")
        ax.set_title("Sorting Time Comparison")
        ax.legend()
        ax.grid(True, which="both", ls="--", linewidth=0.5)

        plt.tight_layout()
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        fig.savefig(filename.replace(".png", ".pdf"), dpi=300, bbox_inches='tight')
        plt.show()
    plot(df_insertion,df_mergesort,filename="sorting_time_comparison.png", log=True)

    # somewhere between 10 and 100 is the best threshold for insertion sort
    # explore this range more closely

    range_digits = range(10, 501, 5)

    if not os.path.exists("sorting_time_comparison_close.png"):
        insertion_times_close, mergesort_times_close = do_tests(range_digits, N_TESTS)
        #save
        df_insertion_close = pd.DataFrame(insertion_times_close).T
        df_mergesort_close = pd.DataFrame(mergesort_times_close).T
        df_insertion_close.to_csv("insertion_times_close.csv")
        df_mergesort_close.to_csv("mergesort_times_close.csv")
    else:
        print("Loading previous results...")
        df_insertion_close = pd.read_csv("insertion_times_close.csv", index_col=0)
        df_mergesort_close = pd.read_csv("mergesort_times_close.csv", index_col=0)
    # Plotting the results
    plot(df_insertion_close,df_mergesort_close,filename="sorting_time_comparison_close.png", log=False,figsize=(14, 6))

    # Based on the plots, we chose 120 as the threshold for insertion sort.

    # Verify with varying the threshold
    thresholds = range(0, 201, 4)
    data = []
    N_TESTS=10
    range_digits = range(10, 251, 5)
    for thr in thresholds:
        if not os.path.exists(f"mergesort_times_thr-{thr}.csv"):
            mergesort_times = do_tests(range_digits, N_TESTS, thr)
            mergesort_times = pd.DataFrame(mergesort_times).T
            mergesort_times['threshold'] = thr
            mergesort_times.to_csv(f"mergesort_times_thr-{thr}.csv")
        else:
            print(f"Loading previous results for threshold {thr}...")
            mergesort_times = pd.read_csv(f"mergesort_times_thr-{thr}.csv", index_col=0)
        data.append(mergesort_times)
    
    for df in data:
        df['n_digits'] = df.index.astype(int)
    df_thresholds = pd.concat(data, ignore_index=True)
    experiment_cols = df_thresholds.columns.tolist()
    experiment_cols = [x for x in experiment_cols if isinstance(x, int) or (isinstance(x, str) and x.isdigit())]
    # make each of the experiment_cols a different row
    df_thresholds = pd.melt(df_thresholds, id_vars=['threshold', 'n_digits'], value_vars=experiment_cols, var_name='experiment', value_name='time')
    df_thresholds['experiment'] = df_thresholds['experiment'].astype(int)
    # drop where n_digits <= threshold
    #df_thresholds = df_thresholds[df_thresholds['n_digits'] > df_thresholds['threshold']]
    # calculate mean and std across ignoring experiment and across n_digits
    df_thresholds_mean = df_thresholds.groupby(['threshold']).agg({'time': ['mean', 'std']}).reset_index()
    df_thresholds_mean.columns = ['threshold', 'mean_time', 'std_time']

    # plot the mean and std with points with error bars
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(
        df_thresholds_mean['threshold'], 
        df_thresholds_mean['mean_time'], 
        yerr=df_thresholds_mean['std_time'], 
        fmt='o-', 
        capsize=5
    )
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Time (s)")
    ax.set_title("Threshold vs Time")
    ax.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    fig.savefig("threshold_vs_time.png", dpi=300, bbox_inches='tight')
    plt.show()

# Fonction à compléter / function to complete:
def solve(array) :
    hybrid_sort(array, 0, len(array) - 1, 3)
    return array
# Ne pas modifier le code ci-dessous :
# Do not modify the code below :

def process_numbers(input_file):
    try:
        # Read integers from the input file
        with open(input_file, "r") as f:
            lines = f.readlines() 
            array = list(map(int, lines[0].split()))  # valeur de chaque noeud  

        return solve(array)
    
    except Exception as e:
        print(f"Error: {e}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python tri_hybride.py <input_file>")
        return

    input_file = sys.argv[1]

    print(f"Input File: {input_file}")
    res = process_numbers(input_file)
    print(f"Result: {res}")

if __name__ == "__main__":
    main()
