# Unsupervised Learning in Python

## 1. Clustering for dataset exploration
> * how to discover the underlying groups (or "clusters") in a dataset.

### Unsupervised learning
> #### Clustering 2D points
> > *  create a KMeans model to find 3 clusters, and fit it to the data points from the previous exercise
> > * obtain the cluster labels for some new points using the .predict()

```python
# Import KMeans
from sklearn.cluster import KMeans

# Create a KMeans instance with 3 clusters: model
model = KMeans(n_clusters=3)

# Fit model to points
model.fit(points)

# Determine the cluster labels of new_points: labels
labels = model.predict(new_points)

# Print cluster labels of new_points
print(labels)
```
![86](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/86.png)

> #### Inspect your clustering

```python
# Import pyplot
import matplotlib.pyplot as plt

# Assign the columns of new_points: xs and ys
xs = new_points[:,0]
ys = new_points[:,1]

# Make a scatter plot of xs and ys, using labels to define the colors
plt.scatter(xs, ys, c=labels,alpha=0.5)

# Assign the cluster centers: centroids
centroids = model.cluster_centers_

# Assign the columns of centroids: centroids_x, centroids_y
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]

# Make a scatter plot of centroids_x and centroids_y
plt.scatter(centroids_x, centroids_y, marker='D',s=50)
plt.show()
```
![87](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/87.png)

### Evaluating a clustering
> #### How many clusters of grain?
> > * learned how to choose a good number of clusters for a dataset using the k-means inertia graph

```python
ks = range(1, 6)
inertias = []

for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    
    # Fit model to samples
    model.fit(samples)
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()
```

![88](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/88.png)

> > * The inertia decreases very slowly from 3 clusters to 4, so it looks like 3 clusters would be a good choice for this data

> #### Evaluating the grain clustering
> > * cluster the grain samples into three clusters, and compare the clusters to the grain varieties using a cross-tabulation.

```python
# Create a KMeans model with 3 clusters: model
model = KMeans(n_clusters=3)

# Use fit_predict to fit model and obtain cluster labels: labels
labels = model.fit_predict(samples)

# Create a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'labels': labels, 'varieties': varieties})

# Create crosstab: ct
ct = pd.crosstab(df['labels'],df['varieties'])

# Display ct
print(ct)

```
![89](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/89.png)

> > * The cross-tabulation shows that the 3 varieties of grain separate really well into 3 clusters. 
> > * But depending on the type of data you are working with, the clustering may not always be this good.

### Transforming features for better clusterings

> #### Scaling fish data for clustering
> > * The measurements, such as weight in grams, length in centimeters, and the percentage ratio of height to length, have very different scales. 
> > * In order to cluster this data effectively, you'll need to standardize these features first

```python
# Perform the necessary imports
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Create scaler: scaler
scaler = StandardScaler()

# Create KMeans instance: kmeans
kmeans = KMeans(n_clusters=4)

# Create pipeline: pipeline
pipel

## 2. Visualization with hierarchical clustering and t-SNE
> * learn about two unsupervised learning techniques for data visualization, hierarchical clustering and t-SNE. 
> * Hierarchical clustering merges the data samples into ever-coarser clusters, yielding a tree visualization of the resulting cluster hierarchy. 
> * t-SNE maps the data samples into 2d space so that the proximity of the samples to one another can be visualized.

ine = make_pipeline(scaler, kmeans)
```

> #### Clustering the fish data

```python
# Import pandas
import pandas as pd

# Fit the pipeline to samples
pipeline.fit(samples)

# Calculate the cluster labels: labels
labels = pipeline.predict(samples)

# Create a DataFrame with labels and species as columns: df
df = pd.DataFrame({'labels': labels, 'species': species})

# Create crosstab: ct
ct = pd.crosstab(df['labels'], df['species'])

# Display ct
print(ct)
```
![90](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/90.png)

> #### Clustering stocks using KMeans
> > * Normalizer() is different to StandardScaler(), which you used in the previous exercise. 
> > * While StandardScaler() standardizes features (such as the features of the fish data from the previous exercise) by removing the mean and scaling to unit variance, 
> > * Normalizer() rescales each sample - here, each company's stock price - independently of the other.

```python
# Import Normalizer
from sklearn.preprocessing import Normalizer

# Create a normalizer: normalizer
normalizer = Normalizer()

# Create a KMeans model with 10 clusters: kmeans
kmeans = KMeans(n_clusters=10)

# Make a pipeline chaining normalizer and kmeans: pipeline
pipeline = make_pipeline(normalizer, kmeans)

# Fit pipeline to the daily price movements
pipeline.fit(movements)
```
> #### Which stocks move together?
> > * which company have stock prices that tend to change in the same way

```python
# Import pandas
import pandas as pd

# Predict the cluster labels: labels
labels = pipeline.predict(movements)

# Create a DataFrame aligning labels and companies: df
df = pd.DataFrame({'labels': labels, 'companies': companies})

# Display df sorted by cluster label
print(df.sort_values('labels'))
```
![91](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/91.png)

## 2. Visualization with hierarchical Cluster
> * learn about two unsupervised learning techniques for data visualization, hierarchical clustering and t-SNE. Hierarchical clustering 

### Visualizing hierarchies
> #### Hierarchical clustering of the grain data
> > * the SciPy linkage() function performs hierarchical clustering on an array of samples. 
> > * Use the linkage() function to obtain a hierarchical clustering of the grain samples, and use dendrogram() to visualize the result. 

> * Dimension reduction summarizes a dataset using its common occuring patterns
> * learn about the most fundamental of dimension reduction techniques, "Principal Component Analysis" ("PCA"). 
> * PCA is often used before supervised learning to improve model performance and generalization. It can also be useful for unsupervised learning. 

```python
# Perform the necessary imports
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# Calculate the linkage: mergings
mergings = linkage(samples, method='complete')

# Plot the dendrogram, using varieties as labels
dendrogram(mergings,
           labels=varieties,
           leaf_rotation=90,
           leaf_font_size=6,
)
plt.show()
```
![92](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/92.png)

> #### Hierarchies of stocks
> > *  you'll need to use the normalize() function from sklearn.preprocessing instead of Normalizer.

```python
# Import normalize
from sklearn.preprocessing import normalize

# Normalize the movements: normalized_movements
normalized_movements = normalize(movements)

# Calculate the linkage: mergings
mergings = linkage(normalized_movements, method='complete')

# Plot the dendrogram
dendrogram(mergings, labels=companies, leaf_rotation=90,leaf_font_size=6)
plt.show()
```
![93](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/93.png)

### Cluster labels in hierarchical clustering
> #### Which clusters are closest?
> > * In complete linkage, the distance between clusters is the distance between the furthest points of the clusters. 
> > * In single linkage, the distance between clusters is the distance between the closest points of the clusters.
> > * In single linkage, cluster 3 is the closest to cluster 2.
> > * In complete linkage, cluster 1 is the closest to cluster 2

![94](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/94.png)

> #### Different linkage, different hierarchical clustering!
> > * a hierarchical clustering of the voting countries at the Eurovision song contest using 'complete' linkage. Now, perform a hierarchical clustering of the voting countries with 'single' linkage

```python
# Perform the necessary imports
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

# Calculate the linkage: mergings
mergings = linkage(samples,method='single')

# Plot the dendrogram
dendrogram(mergings, labels=country_names, leaf_rotation=90, leaf_font_size=6)
plt.show()
```

![95](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/95.png)

> #### Extracting the cluster labels
> > * use the fcluster() function to extract the cluster labels for this intermediate clustering, 
> > * and compare the labels with the grain varieties using a cross-tabulation.

```python
# Perform the necessary imports
import pandas as pd
from scipy.cluster.hierarchy import fcluster

# Use fcluster to extract labels: labels
labels = fcluster(mergings,6,criterion='distance')

# Create a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'labels': labels, 'varieties': varieties})

# Create crosstab: ct
ct = pd.crosstab(df['labels'], df['varieties'])

# Display ct
print(ct)
```

![96](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/96.png)

### t-SNE for 2-dimensional maps
> #### t-SNE visualization of grain dataset
> > * t-SNE visualization manages to separate the 3 varieties of grain samples.

```python
# Import TSNE
from sklearn.manifold import TSNE

# Create a TSNE instance: model
model = TSNE(learning_rate=200)

# Apply fit_transform to samples: tsne_features
tsne_features = model.fit_transform(samples)

# Select the 0th feature: xs
xs = tsne_features[:,0]

# Select the 1st feature: ys
ys = tsne_features[:,1]

# Scatter plot, coloring by variety_numbers
plt.scatter(xs,ys,c=variety_numbers)
plt.show()
```
![97](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/97.png)

> #### A t-SNE map of the stock market
> > * The stock price movements for each company are available as the array normalized_movements (these have already been normalized for you). The list companies

```python
# Import TSNE
from sklearn.manifold import TSNE

# Create a TSNE instance: model
model = TSNE(learning_rate=50)

# Apply fit_transform to normalized_movements: tsne_features
tsne_features = model.fit_transform(normalized_movements)

# Select the 0th feature: xs
xs = tsne_features[:,0]

# Select the 1th feature: ys
ys = tsne_features[:,1]

# Scatter plot
plt.scatter(xs, ys, alpha=0.5)

# Annotate the points
for x, y, company in zip(xs, ys, companies):
    plt.annotate(company, (x, y), fontsize=5, alpha=0.75)
plt.show()
```
![98](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/98.png)

> * It's visualizations such as this that make t-SNE such a powerful tool for extracting quick insights from high dimensional data

## 3. Decorrelating your data and dimension reduction

### Visualizing the PCA transformation
> #### Correlated data in nature
> > * given an array grains giving the width and length of samples of grain. You suspect that width and length will be correlated
> > * make a scatter plot of width vs length and measure their Pearson correlation.

```python
# Perform the necessary imports
import matplotlib.pyplot as plt
from scipy.stats import  pearsonr

# Assign the 0th column of grains: width
width = grains[:,0]

# Assign the 1st column of grains: length
length = grains[:,1]

# Scatter plot width vs length
plt.scatter(width, length)
plt.axis('equal')
plt.show()

# Calculate the Pearson correlation
correlation, pvalue = pearsonr(width, length)

# Display the correlation
print(correlation) # 0.8604
```
![99](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/99.png)

> #### Decorrelating the grain measurements with PCA
> > * use PCA to decorrelate these measurements, then plot the decorrelated points and measure their Pearson correlation

```python
# Import PCA
from sklearn.decomposition import PCA

# Create PCA instance: model
model = PCA()

# Apply the fit_transform method of model to grains: pca_features
pca_features = model.fit_transform(grains)

# Assign 0th column of pca_features: xs
xs = pca_features[:,0]

# Assign 1st column of pca_features: ys
ys = pca_features[:,1]

# Scatter plot xs vs ys
plt.scatter(xs, ys)
plt.axis('equal')
plt.show()

# Calculate the Pearson correlation of xs and ys
correlation, pvalue = pearsonr(xs, ys)

# Display the correlation
print(correlation) # 0.0
```
![100](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/100.png)

> #### Principal components
> > * the principal components are the directions along which the the data varies.
> > * Both plot 1 and plot 3., the plots could the axes represent the principal components of the point cloud

![101](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/101.png)

### Intrinsic dimension
> #### The first principal component
> > * The first principal component of the data is the direction in which the data varies the most. 

```python
# Make a scatter plot of the untransformed points
plt.scatter(grains[:,0], grains[:,1])

# Create a PCA instance: model
model = PCA()

# Fit model to points
model.fit(grains)

# Get the mean of the grain samples: mean
mean = model.mean_

# Get the first principal component: first_pc
first_pc = model.components_[0,:]

# Plot first_pc as an arrow, starting at mean
plt.arrow(mean[0], mean[1], first_pc[0], first_pc[1], color='red', width=0.01)

# Keep axes on same scale
plt.axis('equal')
plt.show()
```
![102](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/102.png)

> #### Variance of the PCA features
> > * The fish dataset is 6-dimensional. But what is its intrinsic dimension?
> > * Recall that the intrinsic dimension is the number of PCA features with significant variance.

```python
# Perform the necessary imports
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

# Create scaler: scaler
scaler = StandardScaler()

# Create a PCA instance: pca
pca = PCA()

# Create pipeline: pipeline
pipeline = make_pipeline(scaler,pca)

# Fit the pipeline to 'samples'
pipeline.fit(samples)

# Plot the explained variances
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.show()
```
![103](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/103.png)

> > *  It looks like PCA features 0 and 1 have significant variance

### Dimension reduction with PCA
> #### Dimension reduction of the fish measurements
> > * use PCA for dimensionality reduction of the fish measurements, retaining only the 2 most important components

```python
# Import PCA
from sklearn.decomposition import PCA

# Create a PCA model with 2 components: pca
pca = PCA(n_components=2)

# Fit the PCA instance to the scaled samples
pca.fit(scaled_samples)

# Transform the scaled samples: pca_features
pca_features = pca.transform(scaled_samples)

# Print the shape of pca_features
print(pca_features.shape)  #(85, 2) successfully reduced the dimensionality from 6 to 2
```

> #### A tf-idf word-frequency array
> > * use the TfidfVectorizer from sklearn. It transforms a list of documents into a word frequency array, which it outputs as a csr_matrix. It has fit() and transform() methods like other sklearn objects
> > * given a list documents of toy documents about pets.

![104](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/104.png)

```python
# Import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Create a TfidfVectorizer: tfidf
tfidf = TfidfVectorizer() 

# Apply fit_transform to document: csr_mat
csr_mat = tfidf.fit_transform(documents)

# Print result of toarray() method
print(csr_mat.toarray())

# Get the words: words
words = tfidf.get_feature_names()

# Print words
print(words)
```
![105](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/105.png)

> #### Clustering Wikipedia
> > * TruncatedSVD is able to perform PCA on sparse arrays in csr_matrix format, such as word-frequency arrays
> > * Combine your knowledge of TruncatedSVD and k-means to cluster some popular pages from Wikipedia
> > * Create a Pipeline object consisting of a TruncatedSVD followed by KMeans
> > * given an array articles of tf-idf word-frequencies of some popular Wikipedia articles, and a list titles of their titles
> > * Use pipeline to cluster the Wikipedia articles.

```python
# Perform the necessary imports
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline

# Create a TruncatedSVD instance: svd
svd = TruncatedSVD(n_components=50)

# Create a KMeans instance: kmeans
kmeans = KMeans(n_clusters=6)

# Create a pipeline: pipeline
pipeline = make_pipeline(svd,kmeans)

# Import pandas
import pandas as pd

# Fit the pipeline to articles
pipeline.fit(articles)

# Calculate the cluster labels: labels
labels = pipeline.predict(articles)

# Create a DataFrame aligning labels and titles: df
df = pd.DataFrame({'label': labels, 'article': titles})

# Display df sorted by cluster label
print(df.sort_values('label'))
```
![106](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/106.png)

## 4. Discovering interpretable features
> * learn about a dimension reduction technique called "Non-negative matrix factorization" ("NMF") that expresses samples as combinations of interpretable parts
> * For example, it expresses documents as combinations of topics, and images in terms of commonly occurring visual patterns.

### Non-negative matrix factorization (NMF)
> #### NMF applied to Wikipedia articles
> > *  apply NMF, this time using the tf-idf word-frequency array of Wikipedia articles, given as a csr matrix articles

```python
# Import NMF
from sklearn.decomposition import NMF

# Create an NMF instance: model
model = NMF(n_components=6)

# Fit the model to articles
model.fit(articles)

# Transform the articles: nmf_features
nmf_features = model.transform(articles)

# Print the NMF features
print(nmf_features)
```

![107](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/107.png)

> #### NMF features of the Wikipedia articles

```python
# Import pandas
import pandas as pd

# Create a pandas DataFrame: df
df = pd.DataFrame(nmf_features, index=titles)

# Print the row for 'Anne Hathaway'
print(df.loc['Anne Hathaway'])

# Print the row for 'Denzel Washington'
print(df.loc['Denzel Washington'])
```
![108](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/108.png)

> > * Notice that for both actors, the NMF feature 3 has by far the highest value. This means that both articles are reconstructed using mainly the 3rd NMF component.

### NMF learns interpretable parts
> #### NMF learns topics of documents
> > * The NMF model you built earlier is available as model, while words is a list of the words that label the columns of the word-frequency array.

```python
# Import pandas
import pandas as pd

# Create a DataFrame: components_df
components_df = pd.DataFrame(model.components_,columns=words)

# Print the shape of the DataFrame
print(components_df.shape)

# Select row 3: component
component = components_df.iloc[3]

# Print result of nlargest
print(component.nlargest())
```
![109](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/109.png)

> > * recognise the topics that the articles about Anne Hathaway and Denzel Washington have in common! 

> #### NMF learns the parts of images
> > * NMF has expressed the digit as a sum of the components!

```python
# Import NMF
from sklearn.decomposition import NMF

# Create an NMF model: model
model = NMF(n_components=7)

# Apply fit_transform to samples: features
features = model.fit_transform(samples)

# Call show_as_image on each component
for component in model.components_:
    show_as_image(component)

# Assign the 0th row of features: digit_features
digit_features = features[0,:]

# Print digit_features
print(digit_features)
```
> #### PCA doesn't learn parts
> > * Unlike NMF, PCA doesn't learn the parts of things. 
> > * the components of PCA do not represent meaningful parts of images of LED digits

```python
# Import PCA
from sklearn.decomposition import PCA

# Create a PCA instance: model
model = PCA(n_components=7)

# Apply fit_transform to samples: features
features = model.fit_transform(samples)

# Call show_as_image on each component
for component in model.components_:
    show_as_image(component)
```

### Building recommender systems using NMF
> #### Which articles are similar to 'Cristiano Ronaldo'?
> > * Apply this to your NMF model for popular Wikipedia articles, by finding the articles most similar to the article about the footballer Cristiano Ronaldo. 
> > * The NMF features you obtained earlier are available as nmf_features, while titles is a list of the article titles.

```python
# Perform the necessary imports
import pandas as pd
from sklearn.preprocessing import normalize

# Normalize the NMF features: norm_features
norm_features = normalize(nmf_features)

# Create a DataFrame: df
df = pd.DataFrame(norm_features, index=titles)

# Select the row corresponding to 'Cristiano Ronaldo': article
article = df.loc['Cristiano Ronaldo']

# Compute the dot products: similarities
similarities = df.dot(article)

# Display those with the largest cosine similarity
print(similarities.nlargest())
```
![110](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/110.png)

> #### Recommend musical artists
> > * learned about NMF to recommend popular music artists! 
> > * given a sparse array artists whose rows correspond to artists and whose column correspond to users. 
> > * The entries give the number of times each artist was listened to by each user.
> > * build a pipeline and transform the array into normalized NMF features.
> > * MaxAbsScaler, transforms the data so that all users have the same influence on the model, regardless of how many different artists they've listened to.
> > * Suppose you were a big fan of Bruce Springsteen - which other musicial artists might you like?
> > * Use your NMF features from the previous exercise and the cosine similarity to find similar musical artists.

```python
# Perform the necessary imports
from sklearn.decomposition import NMF
from sklearn.preprocessing import Normalizer, MaxAbsScaler
from sklearn.pipeline import make_pipeline

# Create a MaxAbsScaler: scaler
scaler = MaxAbsScaler()

# Create an NMF model: nmf
nmf = NMF(n_components=20)

# Create a Normalizer: normalizer
normalizer = Normalizer()

# Create a pipeline: pipeline
pipeline = make_pipeline(scaler,nmf,normalizer)

# Apply fit_transform to artists: norm_features
norm_features = pipeline.fit_transform(artists)

# Import pandas
import pandas as pd

# Create a DataFrame: df
df = pd.DataFrame(norm_features, index=artist_names)

# Select row of 'Bruce Springsteen': artist
artist = df.loc['Bruce Springsteen']

# Compute cosine similarities: similarities
similarities = df.dot(artist)

# Display those with highest cosine similarity
print(similarities.nlargest())

```

![111](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/111.png)

![112](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/112.png)