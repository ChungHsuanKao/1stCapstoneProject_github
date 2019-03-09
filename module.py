import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    # Compute correlation matrix: corr_mat
    corr_mat = np.corrcoef(x, y)

    # Return entry [0,1]
    return corr_mat[0,1]


def pearson_permutation_plot(dataframe, other_features, target_variable, size=1):
    # Prepare taget and revenue
    target = list(dataframe[target_variable])
    other = list(dataframe[other_features])

    #Calculate correlation
    from Function.module import pearson_r
    r = pearson_r(target, other)

    # Initialize permutation replicates: perm_replicates
    perm_replicates = np.empty(size)

    # Draw replicates
    for i in range(size):
        # Permute x measurments: x_permuted
        target_permuted = np.random.permutation(target)

        # Compute Pearson correlation
        perm_replicates[i] = pearson_r(target_permuted, other)

    # Compute p-value: p
    p = np.sum(perm_replicates >= r) / float(size)
    print('P-value is lower than 0.05: {}'.format(p < 0.05))

    # linear regression
    a, b = np.polyfit(other, target, 1)
    line_x = np.array([min(other), max(other)])
    line_y = line_x * a + b

    # Plot
    plt.subplot(1, 2, 1)
    plt.style.use('ggplot')
    sns.scatterplot(x=other_features, y=target_variable, data=dataframe)
    plt.xticks(rotation=90)
    plt.xlabel(other_features)
    plt.ylabel(target_variable)
    plt.title('Correlation = ' + str(round(r,2)))
    plt.plot(line_x, line_y, color = 'black')

    plt.subplot(1, 2, 2)
    plt.style.use('ggplot')
    plt.hist(perm_replicates)
    plt.axvline(x=r, color='black')
    plt.ylabel('Frequency')
    plt.xlabel('Correlation')
    plt.title('Simulation')

    plt.tight_layout()


def diff_of_means(data_1, data_2):
    """Difference in means of two arrays."""

    # The absluted difference of means of data_1, data_2: diff
    diff = np.mean(data_1) - np.mean(data_2)

    return diff

def bootstrap_replicate_1d(data, func):
    return func(np.random.choice(data, size=len(data)))


def draw_bs_reps(data, func, size):
    """Draw bootstrap replicates."""

    # Initialize array of replicates: bs_replicates
    bs_replicates = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_replicates[i] = bootstrap_replicate_1d(data, func)

    return bs_replicates


def permutation_sample(data1, data2):
    """Generate a permutation sample from two data sets."""

    # Concatenate the data sets: data
    data = np.concatenate((data1, data2))

    # Permute the concatenated array: permuted_data
    permuted_data = np.random.permutation(data)

    # Split the permuted array into two: perm_sample_1, perm_sample_2
    perm_sample_1 = permuted_data[:len(data1)]
    perm_sample_2 = permuted_data[len(data1):]

    return perm_sample_1, perm_sample_2


def draw_perm_reps(data_1, data_2, func, size=1):
    """Generate multiple permutation replicates."""

    # Initialize array of replicates: perm_replicates
    perm_replicates = np.empty(size)

    for i in range(size):
        # Generate permutation sample
        perm_sample_1, perm_sample_2 = permutation_sample(data_1, data_2)

        # Compute the test statistic
        perm_replicates[i] = func(perm_sample_1, perm_sample_2)

    return perm_replicates


def mean_diff_testing(dataframe, feature, target, size=1):
    # The revenue of two different groups in the variable
    group_a = dataframe[dataframe[feature] == 0][target].values
    group_b = dataframe[dataframe[feature] == 1][target].values

    # Compute difference of mean impact force from experiment: empirical_diff_means
    empirical_diff_means = diff_of_means(group_a, group_b)

    # Draw permutation replicates: perm_replicates
    perm_replicates = draw_perm_reps(group_a, group_b, diff_of_means, size)

    # Compute p-value: p
    p = np.sum(abs(perm_replicates) >= abs(empirical_diff_means)) / len(perm_replicates)

    # Plot
    plt.style.use('ggplot')
    plt.hist(perm_replicates)
    plt.axvline(x = empirical_diff_means, color = 'black')
    plt.title('P-value < 0.05: {}'.format(p < 0.05))
    plt.xlabel('Mean difference of revenues in ' + feature)
    plt.ylabel('Frequency')


def mean_diff_p(dataframe, feature, target, size=1):
    # The revenue of two different groups in the variable
    group_a = dataframe[dataframe[feature] == 0][target].values
    group_b = dataframe[dataframe[feature] == 1][target].values

    # Compute difference of mean impact force from experiment: empirical_diff_means
    empirical_diff_means = diff_of_means(group_a, group_b)

    # Draw permutation replicates: perm_replicates
    perm_replicates = draw_perm_reps(group_a, group_b, diff_of_means, size)

    # Compute p-value: p
    p = np.sum(abs(perm_replicates) >= abs(empirical_diff_means)) / len(perm_replicates)
    return p
