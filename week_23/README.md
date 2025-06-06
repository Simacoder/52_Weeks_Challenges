# **Guide to Graphs — From Google Maps to Chessboards**# Guide to Graphs — From Google Maps to Chessboards

Most of us use Google Maps without thinking twice. You open the app, check which route has the least traffic, and hit start. Then somewhere along the way – maybe you miss a turn (I do that often) – and Maps calmly recalculates your route, showing you a new path that still gets you to your destination.

Behind that seamless rerouting is a graph – not a chart, but a structure of nodes (places) and edges (roads) that allows Google Maps to calculate the shortest, fastest, or least congested path from point A to point B.

Once you start noticing them, you’ll realize graphs are everywhere. If you’ve ever used:

* Google Maps to get from one city to another
* LinkedIn to see how you’re connected to someone
* Git to visualize branches and merges

…you’ve interacted with a graph.

Graphs are everywhere, in how we plan routes, recommend friends, manage project dependencies, and even predict the possible moves of a knight on a chessboard. But to use them well, we first need to understand how they’re structured and why they’re so useful.

---

## Table of Contents

* [What is a Graph?](#what-is-a-graph)
* [Types of Graphs](#types-of-graphs)
* [How Graphs Are Represented](#how-graphs-are-represented)
* [Graph Traversal](#graph-traversal)
* [Knight's Travails: A Real-World Graph Problem](#knights-travails-a-real-world-graph-problem)
* [Wrapping Up](#wrapping-up)

---

## Prerequisites

This article is beginner-friendly, and no prior knowledge of graphs is required. To follow along with the code examples, it helps to have:

* Some familiarity with data structures like stacks and queues.
* I use Python for the code snippets, but if you’ve worked with another language, you should be able to follow along easily.

---

## What is a Graph?

At its core, a graph is a collection of nodes (also called vertices) and edges – connections that link those nodes together.

If it sounds simple, that’s because it is. The power of graphs isn’t in their complexity, it’s in their flexibility. You can use them to represent almost anything: people, cities, web pages, tasks, game moves, and the relationships between them.

### Nodes (Vertices)

Each node is a point in the graph. It might represent:

* A location (like a city on a map)
* A person (in a social network)
* A page (for example, on the web)
* A square (like one on a chessboard)

### Edges

An edge is a connection between two nodes. It could represent:

* A road between two cities
* A friendship between two users
* A hyperlink between two web pages
* A legal knight move between two squares

Edges can have direction (one-way or two-way), weight (like distance or cost), or be simple and unweighted.

---

## Types of Graphs

### Directed Graphs

Connections move in a specific direction. Example: Twitter follow relationships.

### Undirected Graphs

Connections go both ways. Example: Facebook friendships.

### Weighted Graphs

Edges have values (like distance, cost, or time). Example: Google Maps routing.

### Unweighted Graphs

Edges are either present or not, with no additional information.

### Cyclic Graphs

Contain cycles — paths where you can return to the starting node.

### Acyclic Graphs

No cycles. Often used in task dependency systems.

### Directed Acyclic Graphs (DAGs)

Edges are directed and no cycles exist. Example: Git commit history, package dependencies.

---

## How Graphs Are Represented

### Adjacency List

Each node maps to a list of its neighbors. Efficient for sparse graphs.

Example:

```python
{
    'A': ['B', 'C'],
    'B': ['A', 'D'],
    'C': ['A', 'D'],
    'D': ['B', 'E'],
    'E': ['D']
}
```

### Adjacency Matrix

A 2D array where rows and columns represent nodes. Each cell indicates the presence (1) or absence (0) of an edge.

Pros: Instant edge lookup.
Cons: High memory use for sparse graphs.

---

## Graph Traversal

Traversal is the process of visiting nodes in a specific order. Two major types:

### Breadth-First Search (BFS)

Uses a queue. Explores all neighbors before going deeper.

```python
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])

    while queue:
        node = queue.popleft()
        if node not in visited:
            print(node)
            visited.add(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    queue.append(neighbor)
```

### Depth-First Search (DFS)

Uses a stack. Dives deep along a path before backtracking.

```python
def dfs(graph, start):
    visited = set()
    stack = [start]

    while stack:
        node = stack.pop()
        if node not in visited:
            print(node)
            visited.add(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    stack.append(neighbor)
```

---

## Knight's Travails: A Real-World Graph Problem

Coming soon: Modeling knight moves on a chessboard using graphs and solving with BFS to find the shortest path.

---

## Wrapping Up

Graphs are a flexible and powerful way to represent relationships and movement. From social networks to transportation systems and games, understanding how to build and traverse graphs opens the door to solving many interesting and real-world problems.

Next steps? Try implementing a graph yourself and solve a puzzle using BFS or DFS. Maybe even tackle the knight's travails problem!

Happy graphing! 📊

# AUTHOR
- Simanga Mchunu