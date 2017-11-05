# Network Analysis in Python (Part 1)

## 1. Introduction to networks
> * introduced to fundamental concepts in network analytics while becoming acquainted with a real-world Twitter network dataset
> * learn about NetworkX, a library that allows you to manipulate, analyze, and model graph data
> * learn about different types of graphs as well as how to rationally visualize them.

### Introduction to networks
> #### What is a network?

![124](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/124.png)

> #### Basics of NetworkX API, using Twitter network
> > * What is the size of the graph T

![125](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/125.png)

> #### Basic drawing of a network using NetworkX
> > * practice using NetworkX's drawing facilities. It has been pre-loaded as T_sub

```python
# Import necessary modules
import networkx as nx
import matplotlib.pyplot as plt

# Draw the graph to screen
nx.draw(T_sub)
plt.show()
```

![126](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/126.png)

> #### Queries on a graph
> > * The .nodes() method returns a list of nodes
> > * the .edges() method returns a list of tuples, in which each tuple shows the nodes that are present on that edge. 
> > * passing in the keyword argument data=True in these methods retrieves the corresponding metadata associated with the nodes and edges as well.
> > *  a list comprehension: [ output expression for iterator variable in iterable if predicate expression ].

```python
# Use a list comprehension to get the nodes of interest: noi
noi = [n for n, d in T.nodes(data=True) if d['occupation'] == 'scientist']

# Use a list comprehension to get the edges of interest: eoi
eoi = [(u, v) for u, v, d in T.edges(data=True) if d['date'] < date(2010, 1, 1)]
```

### Types of graphs
> #### Checking the un/directed status of a graph
> > * Use Python's built-in type() function

![127](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/127.png)

> #### Specifying a weight on edges
> > * Weights can be added to edges in a graph, typically indicating the "strength" of an edge. 
> > * In NetworkX, the weight is indicated by the 'weight' key in the metadata dictionary. 

```python
# Set the weight of the edge
T[1][10]['weight'] = 2

# Iterate over all the edges (with metadata)
for u, v, d in T.edges(data=True):

    # Check if node 293 is involved
    if 293 in [u,v]:
    
        # Set the weight to 1.1
        T[u][v]['weight'] = 1.1
```

> #### Checking whether there are self-loops in the graph
> > * allows edges that begin and end on the same node; 
> > * this would be non-intuitive for a social network graph, it is useful to model data such as trip networks, in which individuals begin at one location and end in another.
> > * NetworkX graphs provide a method for this purpose: .number_of_selfloops().

```python
# Define find_selfloop_nodes()
def find_selfloop_nodes(G):
    """
    Finds all nodes that have self-loops in the graph G.
    """
    nodes_in_selfloops = []
    
    # Iterate over all the edges of G
    for u, v in G.edges():
    
    # Check if node u and node v are the same
        if u == v:
        
            # Append node u to nodes_in_selfloops
            nodes_in_selfloops.append(u)
            
    return nodes_in_selfloops

# Check whether number of self loops equals the number of nodes in self loops
assert T.number_of_selfloops() == len(find_selfloop_nodes(T))
```

### Network visualization
> #### Visualizing using Matrix plots
> > * nxviz provides a MatrixPlot object.

![128](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/128.png)

```python
# Import nxviz
import nxviz as nv

# Create the MatrixPlot object: m
m = nv.MatrixPlot(T)

# Draw m to the screen
m.draw()

# Display the plot
plt.show()

# Convert T to a matrix format: A
A = nx.to_numpy_matrix(T)

# Convert A back to the NetworkX form as a directed graph: T_conv
T_conv = nx.from_numpy_matrix(A, create_using=nx.DiGraph())

# Check that the `category` metadata field is lost from each node
for n, d in T_conv.nodes(data=True):
    assert 'category' not in d.keys()
```

> #### Visualizing using Circos plots
> > * Circos plots are a rational, non-cluttered way of visualizing graph data, in which nodes are ordered around the circumference in some fashion, and the edges are drawn within the circle that results

![129](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/129.png)

```python
# Import necessary modules
import matplotlib.pyplot as plt
from nxviz import CircosPlot

# Create the CircosPlot object: c
c = CircosPlot(T)

# Draw c to the screen
c.draw()

# Display the plot
plt.show()
```

## 2. Important nodes
> * learn about ways of identifying nodes that are important in a network
> * learn the basics of path-finding algorithms
> * deep dive into the Twitter network dataset which will reinforce the concepts you've learned, such as degree centrality and betweenness centrality.

### Degree centrality
> #### Compute number of neighbors for each node
> > * How do you evaluate whether a node is an important one or not?
> > * the number of neighbors that a node has.
> > * Every NetworkX graph G exposes a .neighbors(n) method that returns a list of nodes that are the neighbors of the node n
> > * write a function that returns all nodes that have m neighbors

```python
# Define nodes_with_m_nbrs()
def nodes_with_m_nbrs(G, m):
    """
    Returns all nodes in graph G that have m neighbors.
    """
    nodes = set()
    
    # Iterate over all nodes in G
    for n in G.nodes():
    
        # Check if the number of neighbors of n matches m
        if len(G.neighbors(n)) == m:
        
            # Add the node n to the set
            nodes.add(n)
            
    # Return the nodes with m neighbors
    return nodes

# Compute and print all nodes in T that have 6 neighbors
six_nbrs = nodes_with_m_nbrs(T, 6)
print(six_nbrs)

```
> #### Compute degree distribution
> > * The number of neighbors that a node has is called its "degree"
> > * it's possible to compute the degree distribution across the entire graph

```python
# Compute the degree of every node: degrees
degrees = [len(T.neighbors(n)) for n in T.nodes()]

# Print the degrees
print(degrees)
```

> #### Degree centrality distribution
> > * The degree centrality is the number of neighbors divided by all possible neighbors that it could have
> > * Depending on whether self-loops are allowed, the set of possible neighbors a node could have could also include the node itself
> > * The nx.degree_centrality(G) function returns a dictionary, where the keys are the nodes and the values are their degree centrality values

```python
# Import matplotlib.pyplot
import matplotlib.pyplot as plt

# Compute the degree centrality of the Twitter network: deg_cent
deg_cent = nx.degree_centrality(T)

# Plot a histogram of the degree centrality distribution of the graph.
plt.figure()
plt.hist(list(deg_cent.values()))
plt.show()

# Plot a histogram of the degree distribution of the graph
plt.figure()
plt.hist(degrees)
plt.show()

# Plot a scatter plot of the centrality distribution and the degree distribution
plt.figure()
plt.scatter(degrees,list(deg_cent.values()))
plt.show()
```
![130](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/130.png)

![131](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/131.png)

![132](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/132.png)

> > * it should not surprise you to see a perfect correlation between the centrality distribution and the degree distribution


### Graph algorithms
> #### breadth-first search
> > * the "breadth-first search" (BFS) algorithm
> > * Pathfinding algorithms are important because they provide another way of assessing node importance
> > * going to build up slowly to get to the final BFS algorithm. 

```python
# Define path_exists()
def path_exists(G, node1, node2):
    """
    This function checks whether a path exists between two nodes (node1, node2) in graph G.
    """
    visited_nodes = set()
    
    # Initialize the queue of cells to visit with the first node: queue
    queue = [node1]  
    
    # Iterate over the nodes in the queue
    for node in queue:
    
        # Get neighbors of the node
        neighbors = G.neighbors(node) 
        
        # Check to see if the destination node is in the set of neighbors
        if node2 in neighbors:
            print('Path exists between nodes {0} and {1}'.format(node1, node2))
            return True
            break
        else:
           # Add current node to visited nodes
           visited_nodes.add(node)
            
           # Add neighbors of current node that have not yet been visited
           queue.extend([n for n in neighbors if n not in visited_nodes])
           
       
       # Check to see if the final element of the queue has been reached
       if node == queue[-1]:
            print('Path does not exist between nodes {0} and {1}'.format(node1, node2))

            # Place the appropriate return statement
            return False
 
 
 
```

### Betweenness centrality
> #### NetworkX betweenness centrality on a social network
> > * Betweenness centrality is a node importance metric that uses information about the shortest paths in a network
> > * NetworkX provides the nx.betweenness_centrality(G) function for computing the betweenness centrality of every node in a graph, and it returns a dictionary where the keys are the nodes and the values are their betweenness centrality measures.

![133](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/133.png)

![134](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/134.png)

```python
# Compute the betweenness centrality of T: bet_cen
bet_cen = nx.betweenness_centrality(T)

# Compute the degree centrality of T: deg_cen
deg_cen = nx.degree_centrality(T)

# Create a scatter plot of betweenness centrality and degree centrality
plt.scatter(list(bet_cen.values()), list(deg_cen.values()))

# Display the plot
plt.show()
```

![135](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/135.png)

> #### Deep dive - Twitter network
> > * First, you're going to find the nodes that can broadcast messages very efficiently to lots of people one degree of separation away

```python
# Define find_nodes_with_highest_deg_cent()
def find_nodes_with_highest_deg_cent(G):

    # Compute the degree centrality of G: deg_cent
    deg_cent = nx.degree_centrality(G)
    
    # Compute the maximum degree centrality: max_dc
    max_dc = max(list(deg_cent.values()))
    
    nodes = set()
    
    # Iterate over the degree centrality dictionary
    for k, v in deg_cent.items():
    
        # Check if the current value has the maximum degree centrality
        if v == max_dc:
        
            # Add the current node to the set of nodes
            nodes.add(k)
            
    return nodes
    
# Find the node(s) that has the highest degree centrality in T: top_dc
top_dc = find_nodes_with_highest_deg_cent(T)
print(top_dc)

# Write the assertion statement
for node in top_dc:
    assert nx.degree_centrality(T)[node] == max(nx.degree_centrality(T).values())
```
> #### Deep dive - Twitter network part II
> > * next, you're going to do an analogous deep dive on betweenness centrality! 

```python
# Define find_node_with_highest_bet_cent()
def find_node_with_highest_bet_cent(G):

    # Compute betweenness centrality: bet_cent
    bet_cent = nx.betweenness_centrality(G)
    
    # Compute maximum betweenness centrality: max_bc
    max_bc = max(list(bet_cent.values()))
    
    nodes = set()
    
    # Iterate over the betweenness centrality dictionary
    for k, v in bet_cent.items():
    
        # Check if the current value has the maximum betweenness centrality
        if v == max_bc:
        
            # Add the current node to the set of nodes
            nodes.add(k)
            
    return nodes

# Use that function to find the node(s) that has the highest betweenness centrality in the network: top_bc
top_bc = find_node_with_highest_bet_cent(T)

# Write an assertion statement that checks that the node(s) is/are correctly identified.
for node in top_bc:
    assert nx.betweenness_centrality(T)[node] == max(nx.betweenness_centrality(T).values())
```

## 3. Structures
> * finding interesting structures within network data
> * essential concepts such as cliques, communities, and subgraphs

### Cliques & communities
> #### Identifying triangle relationships
> > * We may be interested in triangles because they're the simplest complex clique.
> > * In the Twitter network, each node has an 'occupation' label associated with it, in which the Twitter user's work occupation is divided into celebrity, politician and scientist. One potential application of triangle-finding algorithms is to find out whether users that have similar occupations are more likely to be in a clique with one another.

```python
from itertools import combinations

# Define is_in_triangle() 
def is_in_triangle(G, n):
    """
    Checks whether a node `n` in graph `G` is in a triangle relationship or not. 
    
    Returns a boolean.
    """
    in_triangle = False
    
    # Iterate over all possible triangle relationship combinations
    for n1, n2 in combinations(G.neighbors(n), 2):
    
        # Check if an edge exists between n1 and n2
        if G.has_edge(n1, n2):
            in_triangle = True
            break
    return in_triangle
```

> #### Finding nodes involved in triangles
> > * NetworkX provides an API for counting the number of triangles that every node is involved in: nx.triangles(G). It returns a dictionary of nodes as the keys and number of triangles as the values. 

```python
from itertools import combinations

# Write a function that identifies all nodes in a triangle relationship with a given node.
def nodes_in_triangle(G, n):
    """
    Returns the nodes in a graph `G` that are involved in a triangle relationship with the node `n`.
    """
    triangle_nodes = set([n])
    
    # Iterate over all possible triangle relationship combinations
    for n1, n2 in combinations(G.neighbors(n), 2):
    
        # Check if n1 and n2 have an edge between them
        if G.has_edge(n1, n2):
        
            # Add n1 to triangle_nodes
            triangle_nodes.add(n1)
            
            # Add n2 to triangle_nodes
            triangle_nodes.add(n2)

    return triangle_nodes
    
# Write the assertion statement
assert len(nodes_in_triangle(T, 1)) == 35
```
> #### Finding open triangles
> > * they form the basis of friend recommendation systems; if "A" knows "B" and "A" knows "C", then it's probable that "B" also knows "C".

```python
from itertools import combinations

# Define node_in_open_triangle()
def node_in_open_triangle(G, n):
    """
    Checks whether pairs of neighbors of node `n` in graph `G` are in an 'open triangle' relationship with node `n`.
    """
    in_open_triangle = False
    
    # Iterate over all possible triangle relationship combinations
    for n1, n2 in combinations(G.neighbors(n), 2):
    
        # Check if n1 and n2 do NOT have an edge between them
        if not G.has_edge(n1, n2):
        
            in_open_triangle = True
            
            break
            
    return in_open_triangle

# Compute the number of open triangles in T
num_open_triangles = 0

# Iterate over all the nodes in T
for n in T.nodes():

    # Check if the current node is in an open triangle
    if node_in_open_triangle(T, n):
    
        # Increment num_open_triangles
        num_open_triangles += 1
        
print(num_open_triangles)
```

### Maximal cliques
> #### Finding all maximal cliques of size "n"
> > * Maximal cliques are cliques that cannot be extended by adding an adjacent edge, and are a useful property of the graph when finding communities. 
> > * NetworkX provides a function that allows you to identify the nodes involved in each maximal clique in a graph: nx.find_cliques(G)

```python
# Define maximal_cliques()
def maximal_cliques(G, size):
    """
    Finds all maximal cliques in graph `G` that are of size `size`.
    """
    mcs = []
    for clique in nx.find_cliques(G):
        if len(clique) == size:
            mcs.append(clique)
    return mcs

# Check that there are 33 maximal cliques of size 3 in the graph T
assert len(maximal_cliques(T, 3)) == 33
```

### Subgraphs
> #### Subgraphs I
> > * you just want to analyze a subset of nodes in a network. To do so,
> > * you can copy them out into another graph object using G.subgraph(nodes), which returns a new graph object (of the same type as the original graph) that is comprised of the iterable of nodes that was passed in.
> > * A subsampled version of the Twitter network has been loaded as `T`.
T has been converted to an undirected graph.

```python
# Define get_nodes_and_nbrs()
def get_nodes_and_nbrs(G, nodes_of_interest):
    """
    Returns a subgraph of the graph `G` with only the `nodes_of_interest` and their neighbors.
    """
    nodes_to_draw = []
    
    # Iterate over the nodes of interest
    for n in nodes_of_interest:
    
        # Append the nodes of interest to nodes_to_draw
        nodes_to_draw.append(n)
        
        # Iterate over all the neighbors of node n
        for nbr in G.neighbors(n):
        
            # Append the neighbors of n to nodes_to_draw
            nodes_to_draw.append(nbr)
            
    return G.subgraph(nodes_to_draw)

# Extract the subgraph with the nodes of interest: T_draw
T_draw = get_nodes_and_nbrs(T, nodes_of_interest)

# Draw the subgraph to the screen
nx.draw(T_draw)
plt.show()
```
![136](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/136.png)

> #### Subgraphs II
> > * you extract nodes that have a particular metadata property and their neighbors

```python
# Extract the nodes of interest: nodes
nodes = [n for n, d in T.nodes(data=True) if d['occupation'] == 'celebrity']

# Create the set of nodes: nodeset
nodeset = set(nodes)

# Iterate over nodes
for n in nodes:

    # Compute the neighbors of n: nbrs
    nbrs = T.neighbors(n)
    
    # Compute the union of nodeset and nbrs: nodeset
    nodeset = nodeset.union(nbrs)

# Compute the subgraph using nodeset: T_sub
T_sub = T.subgraph(nodeset)

# Draw T_sub to the screen
nx.draw(T_sub)
plt.show()
```
![137](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/137.png)

## 4. Bringing it all together
> * you'll have developed your very own recommendation system which suggests GitHub users who should collaborate together. 

### Case study!
> #### Characterizing the network (I)
> > * use the functions len(G.nodes()) and len(G.edges()) to calculate the number of nodes and edges respectively.

> #### Characterizing the network (II)
> > * node importances, by plotting the degree distribution of a network. 

```python
# Import necessary modules
import matplotlib.pyplot as plt
import networkx as nx 

# Plot the degree distribution of the GitHub collaboration network
plt.hist(list(nx.degree_centrality(G).values()))
plt.show()
```
![138](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/138.png)

> #### Characterizing the network (III)

```python
# Import necessary modules
# Import necessary modules
import matplotlib.pyplot as plt
import networkx as nx

# Plot the degree distribution of the GitHub collaboration network
plt.hist(list(nx.betweenness_centrality(G).values()))
plt.show()
```

### Case study part II: Visualization
> #### MatrixPlot
> > * the matrix is the representation of the edges
> > * nodes are the rows and columns of the matrix, and cells are filled in according to whether an edge exists between the pairs of nodes.
> > * Python's built-in sorted() function takes an iterable and returns a sorted list (in ascending order, by default). Therefore, to access the largest connected component subgraph, the statement is sliced with [-1].

```python
# Import necessary modules
from nxviz import MatrixPlot
import matplotlib.pyplot as plt

# Calculate the largest connected component subgraph: largest_ccs
largest_ccs = sorted(nx.connected_component_subgraphs(G), key=lambda x: len(x))[-1]

# Create the customized MatrixPlot object: h
h = MatrixPlot(graph=largest_ccs, node_grouping='grouping')

# Draw the MatrixPlot to the screen
h.draw()
plt.show()
```
![139](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/139.png)

> #### ArcPlot

```python
# Import necessary modules
from nxviz.plots import ArcPlot
import matplotlib.pyplot as plt

# Iterate over all the nodes in G, including the metadata
for n, d in G.nodes(data=True):

    # Calculate the degree of each node: G.node[n]['degree']
    G.node[n]['degree'] = nx.degree(G, n)
    
# Create the ArcPlot object: a
a = ArcPlot(graph=G, node_order='degree')

# Draw the ArcPlot to the screen
a.draw()
plt.show()
```
![140](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/140.png)

> #### CircosPlot

```python
# Import necessary modules
from nxviz import CircosPlot
import matplotlib.pyplot as plt 
 
# Iterate over all the nodes, including the metadata
for n, d in G.nodes(data=True):

    # Calculate the degree of each node: G.node[n]['degree']
    G.node[n]['degree'] = nx.degree(G, n)

# Create the CircosPlot object: c
c = CircosPlot(graph=G, node_order='degree', node_grouping='grouping', node_color='grouping')

# Draw the CircosPlot object to the screen
c.draw()
plt.show()
```
![141](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/141.png)


### Case study part III: Cliques
> #### Finding cliques (I)
> > * cliques are "groups of nodes that are fully connected to one another"
> > * a maximal clique is a clique that cannot be extended by adding another node in the graph

```python
# Calculate the maximal cliques in G: cliques
cliques = nx.find_cliques(G)

# Count and print the number of maximal cliques in G
print(len(list(cliques)))
```
> #### Finding cliques (II)

```python
# Import necessary modules
import networkx as nx
from nxviz import CircosPlot
import matplotlib.pyplot as plt

# Find the author(s) that are part of the largest maximal clique: largest_clique
largest_clique = sorted(nx.find_cliques(G), key=lambda x:len(x))[-1]

# Create the subgraph of the largest_clique: G_lc
G_lc = G.subgraph(largest_clique)

# Create the CircosPlot object: c
c = CircosPlot(G_lc)

# Draw the CircosPlot to the screen
c.draw()
plt.show()
```
![142](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/142.png)

>  * The subgraph consisting of the largest maximal clique has 38 users

### Case study part IV: Final tasks
> #### Finding important collaborators
> > * use of the degree_centrality() and betweenness_centrality() functions in NetworkX to compute each of the respective centrality scores, and then use that information to find the "important nodes". 

```python
# Compute the degree centralities of G: deg_cent
deg_cent = nx.degree_centrality(G)

# Compute the maximum degree centrality: max_dc
max_dc = max(deg_cent.values())

# Find the user(s) that have collaborated the most: prolific_collaborators
prolific_collaborators = [n for n, dc in deg_cent.items() if dc == max_dc]

# Print the most prolific collaborator(s)
print(prolific_collaborators) 
```
> #### Characterizing editing communities
> > * using the BFS algorithm and concept of maximal cliques to visualize the network with an ArcPlot.
> > * The largest maximal clique in the Github user collaboration network has been assigned to the subgraph G_lmc

```python
# Import necessary modules
from nxviz import ArcPlot
import matplotlib.pyplot as plt
 
# Identify the largest maximal clique: largest_max_clique
largest_max_clique = set(sorted(nx.find_cliques(G), key=lambda x: len(x))[-1])

# Create a subgraph from the largest_max_clique: G_lmc
G_lmc = G.subgraph(largest_max_clique)

# Go out 1 degree of separation
for node in G_lmc.nodes():
    G_lmc.add_nodes_from(G.neighbors(node))
    G_lmc.add_edges_from(zip([node]*len(G.neighbors(node)), G.neighbors(node)))

# Record each node's degree centrality score
for n in G_lmc.nodes():
    G_lmc.node[n]['degree centrality'] = nx.degree_centrality(G_lmc)[n]
        
# Create the ArcPlot object: a
a = ArcPlot(G_lmc, node_order='degree centrality')

# Draw the ArcPlot to the screen
a.draw()
plt.show()
```
![143](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/143.png)

> #### Recommending co-editors who have yet to edit together
> > * leverage the concept of open triangles to recommend users on GitHub to collaborate!

```python
# Import necessary modules
from itertools import combinations
from collections import defaultdict

# Initialize the defaultdict: recommended
recommended = defaultdict(int)

# Iterate over all the nodes in G
for n, d in G.nodes(data=True):

    # Iterate over all possible triangle relationship combinations
    for n1, n2 in combinations(G.neighbors(n), 2):
    
        # Check whether n1 and n2 do not have an edge
        if not G.has_edge(n1, n2):
        
            # Increment recommended
            recommended[(n1, n2)] += 1

# Identify the top 10 pairs of users
all_counts = sorted(recommended.values())
top10_pairs = [pair for pair, count in recommended.items() if count > all_counts[-10]]
print(top10_pairs)
```
![144](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/144.png)

> > * You've identified pairs of users who should collaborate together