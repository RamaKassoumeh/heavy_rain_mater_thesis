import matplotlib.pyplot as plt
import numpy as np
import rasterio
from scipy import stats
from rasterio.plot import show
from scipy.stats import entropy
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

tif_files = ["data/MSG3-SEVI-MSG15-0100-NA-20210714185415.651000000Z-NA.tif",
             "data/MSG3-SEVI-MSG15-0100-NA-20210714103415.979000000Z-NA.tif"]
time_file=['2021-07-14 18:54','2021-07-14 10:34']

min_values_by_band = {}
max_values_by_band = {}
for i, tif_file in enumerate(tif_files):
    with rasterio.open(tif_file) as src:
        for band_number in range(1, 12):
            band_data = src.read(band_number)
            min_value = band_data.min()
            max_value = band_data.max()
            # Append the minimum value to the list for the current band
            if band_number not in min_values_by_band:
                min_values_by_band[band_number] = []
                max_values_by_band[band_number] = []
            min_values_by_band[band_number].append(min_value)
            max_values_by_band[band_number].append(max_value)

for band in range(1, 12):
    for i in range(len(tif_files)):
        for j in range(i + 1, len(tif_files)):
            with rasterio.open(tif_files[i]) as src1:
                with rasterio.open(tif_files[j]) as src2:
                    band_data1 = src1.read(band)
                    band_data2 = src2.read(band)
                    ks_statistic, p_value = stats.ks_2samp(band_data1.ravel(), band_data2.ravel())
                    np.savetxt('array_2d1.txt', band_data1, delimiter=' ', fmt='%f')
                    np.savetxt('array_2d2.txt', band_data2, delimiter=' ', fmt='%f')
                    # Compute histograms
                    hist_band1, _ = np.histogram(band_data1, bins=50,
                                                 range=(min(min_values_by_band[band]), max(max_values_by_band[band])),
                                                 density=True)
                    hist_band2, _ = np.histogram(band_data2, bins=50,
                                                 range=(min(min_values_by_band[band]), max(max_values_by_band[band])),
                                                 density=True)

                    # Add smoothing to avoid division by zero
                    epsilon = 1e-10
                    hist_band1 += epsilon
                    hist_band2 += epsilon

                    # Normalize the histograms to get probability distributions
                    P = hist_band1 / np.sum(hist_band1)
                    Q = hist_band2 / np.sum(hist_band2)

                    # Compute KL Divergence
                    kl_divergence = np.sum(
                        np.where(Q != 0, np.where(P != 0, P * np.log(P / Q), 0),
                                 0))
                    kl_div = entropy(hist_band1, hist_band2)
                    if p_value >= 0.05:
                        print(f"KS between the two file {i},{j} for {band} band is: {p_value}")
                    if kl_div < 1:
                        print(
                            f"KL Divergence between thee two file {i},{j} for {band} band is:{kl_divergence},{kl_div}")



def calc_stats(datasets):
    # Calculate descriptive statistics
    descriptive_stats = []
    for data in datasets:
        mean = np.mean(data)
        variance = np.var(data)
        skewness = stats.skew(data)
        kurtosis = stats.kurtosis(data)
        descriptive_stats.append((mean, variance, skewness, kurtosis))

    # Print descriptive statistics
    for i, stat in enumerate(descriptive_stats):
        print(f"Dataset {i + 1}: Mean={stat[0]}, Variance={stat[1]}, Skewness={stat[2]}, Kurtosis={stat[3]}")



    # Apply Kolmogorov-Smirnov Test
    ks_results = [stats.kstest(data, 'norm', args=(data.mean(), data.std())) for data in datasets]
    for i, result in enumerate(ks_results):
        print(f"Kolmogorov-Smirnov Test for Dataset {i + 1}: Statistic={result.statistic}, p-value={result.pvalue}")

    # Clustering Analysis (using descriptive statistics)
    X = np.array(descriptive_stats)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    labels = kmeans.labels_
    print("Cluster Labels:", labels)

    # Compute distance metrics (e.g., Wasserstein distance)
    distances = cdist(X, X, metric='euclidean')
    print("Pairwise distances between datasets:\n", distances)

for band in range(1, 12):
    datasets = []
    for i in range(len(tif_files)):
        with rasterio.open(tif_files[i]) as src1:
            band_data1 = src1.read(band)
            datasets.append(band_data1)
    calc_stats(datasets)