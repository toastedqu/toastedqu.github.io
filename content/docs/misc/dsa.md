---
title : "Coding"
description: ""
lead: ""
date: 2020-10-06T08:48:45+00:00
lastmod: 2020-10-06T08:48:45+00:00
draft: false
images: []
weight: 100
---
# Array
| Action        | Time |
|---------------|----|
| access()      | O(1) |
| mutate()      | O(n) (begin/mid)<br>O(1) (end) |
| two pointer   | O(n) |
| binary search | O(logn) |
| merge intervals | O(n) |

## Two Pointer

### Left & Right
Usage: 1d array/string, bi-directional problem

Idea:
1. 1 pointer from the left, 1 pointer from the right.
2. Loop to the middle, depending on the problem.
3. Break when meet.

Tips:
- Sort when necessary!
- Use WHILE loop to go over elems that do not break condition.
- 3-pointer (or more) is always an option.

```python
# 845. Longest Mountain in Array
def longestMountain(self, arr: List[int]) -> int:
    ans = 0
    for i in range(1,len(arr)-1):
        if arr[i-1] < arr[i] > arr[i+1]:                # loop pointer, check if mountain condition is met
            l,r = i-1, i+1                              # init two pointers
            while l>0 and arr[l-1]<arr[l]:              # expand mountain left till condition breaks
                l -= 1
            while r<len(arr)-1 and arr[r]>arr[r+1]:     # expand mountain right till condition breaks
                r += 1
            ans = max(ans, r-l+1)                       # update ans
    return ans
```

&nbsp;

### Slow & Fast
Usage: 1d array/string/, uni-directional problem

Idea:
1. 'fast' pointer for iteration, 'slow' pointer for operation (threshold, comparison, etc.)
2. Move 'slow' only when a certain condition is met/broken, depending on the problem.
3. Finish loop.

Tips:
- Clarify when to / not to move 'slow'.
- For CYCLES: fast = 2x speed, slow = 1x speed.

```python
# 142. Linked List Cycle II
def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
    s,f = head,head
    while f and f.next:     # if broken, no cycle
        f = f.next.next     # move 2x
        s = s.next          # move 1x
        if s == f:          # cycle exists
            h = head        # get head
            while h != s:   # when head meets slow, that's the starting cycle point
                h = h.next
                s = s.next
            return h
    return None
```

&nbsp;

### Sliding Window
Usage: 1d array/string, subarray/substring problem

Idea:
1. Init cache(s) with int/arr/map
2. MIN:
    1. Expand the window until condition is met.
    2. Shrink the window until condition is broken.
3. MAX: 
    1. Expand the window until condition is broken.
    2. Dynamically update cache to track condition. If condition is met again, stop 'start'.

Tips:
- Plan out what to do at every single step in every single case. It's OK to write them all out at the beginning and optimize afterwards. The key is to think carefully.
- Don't overthink over cases that do not matter. "collections" maps contain all keys by default, which come in handy in some cases.

```python
# 76. Minimum Window Substring
def minWindow(self, s: str, t: str) -> str:
    if len(t)>len(s): return ""
    count = collections.Counter(t)
    missing = len(t)
    start = 0    
    min_s,min_e = float('-inf'),float('inf')
    for end in range(len(s)):
        # expand
        if count[s[end]]>0: missing -= 1    # if this key is still available in t, drop missing by 1
        count[s[end]] -= 1                  # reduce the count of a letter, no matter what it is.

        # shrink
        while missing == 0:                 # when we have no missing number, condition is met.
            if min_e-min_s > end-start: min_s,min_e = start,end     # update ans
            # if it's never part of t, it will always have a non-positive count, which won't affect us.
            # if it's part of t, it's reasonable to add it back.
            count[s[start]] += 1

            # only keys that are part of t can have positive counts. in such case, condition is broken.
            if count[s[start]]>0: missing += 1

            start += 1  # loop
    return s[min_s:min_e+1] if min_e<float('inf') else ''
```

&nbsp;

## Binary Search
Credit to [zhijun_liao](https://leetcode.com/problems/find-k-th-smallest-pair-distance/solutions/769705/python-clear-explanation-powerful-ultimate-binary-search-template-solved-many-problems/) for helping me understand binary search a lot better.

Binary Search is an underrated algorithm in LC because it seems so easy to understand: It splits the search space into halves, keeps the half with the target, and repeat till target. However, it is so hard to apply Binary Search in LC questions:
- When to exit loop? (`left` < `right` or `left` <= `right`)?
- How to init boundary `left` and `right`?
- How to update boundary (`left = mid` / `left = mid+1`, `right = mid` / `right = mid-1`)?
- How to define return condition `condition(mid)`?

Usage: **Minimize k s.t. condition(k) is True**
- If we can discover some kind of monotonicity, for example, if `condition(k)` is True then `condition(k + 1)` is True, then binary search.
<!-- Tips:
- After the WHILE loop, 
    - 'l' $\rightarrow$ *arr[0]* / *arr[len(arr)]* (the RIGHT)
    - 'r' $\rightarrow$ *arr[0-1]* / *arr[len(arr)-1]* (the LEFT) -->

```python
def binary_search(nums) -> int:
    # Modifiable 1: design condition
    def condition(mid) -> bool:
        pass

    # Modifiable 2: change init of boundaries (must include all elems)
    l,r = 0,len(nums)

    while l<r:
        mid = l+(r-l)//2
        if condition(mid): r = mid
        else: l = mid+1

    # Modifiable 3: change return value (l = minimal k satisfying condition(k))
    return l
```

&nbsp;

## Merge Intervals

<center>
<img src="/images/dsa/merge_interval.jpg" width=50%/>
</center>

Usage: interval problems

Tips:
- Only 2 conditions for two intervals to overlap: **front.start <= back.end** and **back.start <= front.end**.
- Use MIN/MAX to decide merged *start* / *end* for a more efficient merge.
- Sort by *start* / *end* when necessary.

```python
def merge(self, intervals):
    intervals.sort()                                        # sort first, either by start or end depending on the problem
    merged = []
    for interval in intervals:
        if not merged or merged[-1][1] < interval[0]:       # no overlap condition
            merged.append(interval)
        else:                                               
            merged[-1][1] = max(merged[-1][1], interval[1]) # merge by changing the number of the last interval
    return merged
```

&nbsp;

# Stack & Queue
| Action        | Time |
|---------------|----|
| top() / topleft() | O(1) |
| pop() / popleft() | O(1) |
| append()   | O(1) |

&nbsp;

## Monotonic Stack
Usage: increasing/decreasing trend

Tips:
- Understand clearly what the variable for comparison is. Use stack to store
    - the variable for comparison
    - another variable closely associated with it
- If an element violates the condition, enter a WHILE loop. Continuously update and pop values from stack until
    - this element stays in condition.
    - the stack is EMPTY.
- Else, append element to stack.

```python
# 739. Daily Temperatures
def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
    ans = [0]*len(temperatures)
    s = []  # store indices of monotonically decreasing temps
    for i,t in enumerate(temperatures):
        while s and temperatures[s[-1]] < t:  # if a higher temp is met
            ans[s[-1]] = i-s[-1]              # update ans
            s.pop()                           # pop till no higher temp
        s.append(i)                           # append unanswered index to stack
    return ans
```

&nbsp;

# Linked List
Usage: linked list

Tips:
- Carefully keep track of what your "prev" and "curr" pointers are doing at every single step.
- It involves 3 nodes instead of 2 to make a full reverse.
- Don't be afraid to init bunch of temp nodes. They are all O(1) anyway.

```python
def reverseList(self, head):
    prev, curr = None, head
    while curr:
        temp = curr.next
        curr.next = prev
        prev, curr = curr, temp
    return prev
```

&nbsp;

# Tree

## DFS
Usage: longest/specific search problems

Tips: 
- Choose carefully what you want: pre-order / in-order / post-order.
- Each loop should ONLY focus on
    1) the end/base case
    2) the curr node
- When returning bool, specify both True and False end/base cases.

```python
def dfs_pre(node):
    if is_end_case(): return True
    if break_condition(): return False
    ###### ACTION HERE ######
    ###### ACTION ENDS ######
    dfs_pre(node.left)
    dfs_pre(node.right)

def dfs_in(node):
    if is_end_case(): return True
    if break_condition(): return False
    dfs_in(node.left)
    ###### ACTION HERE ######
    ###### ACTION ENDS ######
    dfs_in(node.right)

def dfs_post(node):
    if is_end_case(): return True
    if break_condition(): return False
    dfs_post(node.left)
    dfs_post(node.right)
    ###### ACTION HERE ######
    ###### ACTION ENDS ######
```

&nbsp;

## BFS
Usage: shortest search problems

Tips: queue/priority queue

Tree with queue
```python
def bfs(node):
    if not node: return
    q = collections.deque([node])
    while q:
        node = q.popleft()
        ###### ACTION HERE ######
        ###### ACTION ENDS ######
        if node.left:  q.append(node.left)
        if node.right: q.append(node.right)
```

&nbsp;

# Heap
Usage: get min/max fast
| Action | Time |
|------|----|
| top()  | O(1) |
| insert()  | O(logn) |
| remove()  | O(logn) |
| heapify() | O(n) |

## Two Heap
Usage: scheduling, median, any problem that involves both min and max somehow.

Tips:
- Set up 2 heaps:
    - small: max heap (i.e., negative min heap)
    - large: min heap
- Use their length as storage condition
- Do NOT pop when looking up items. Use index (0 for root).

```python
# 295. Find Median from Data Stream
class MedianFinder:
    def __init__(self):
        self.small = [] # heap for the smaller half (negative so that min heap works)
        self.large = [] # heap for the larger half

    def addNum(self, num: int) -> None: # O(logn)
        # It doesn't really matter which one has one more value than the other. But be consistent during interview.
        # In this case, we allow "small" to store one more value than "large" when #nums is odd.
        if len(self.small)==len(self.large):                                 # if #nums is now even
            heapq.heappush(self.small, -heapq.heappushpop(self.large, num))  # push new num to "large", pop the smallest from "large", put it in "small"
        else:
            heapq.heappush(self.large, -heapq.heappushpop(self.small, -num)) # push new num to "small", pop the largest from "small", put it in "large"

    def findMedian(self) -> float:      # O(1)
        if len(self.small)==len(self.large):        
            return (self.large[0]-self.small[0])/2
        else:
            return -self.small[0]
```

&nbsp;

# Graph
| Algorithm                | Time         | DS needed                    |
|--------------------------|--------------|------------------------------|
| DFS                      | O(n)         | set (and stack if iteration) |
| BFS                      | O(n)         | set, queue                   |
| Union-Find               | O(nlogn)     | array, tree                  |
| Topological Sort         | O(n)         | array, set, queue            |
| Dijkstra's Shortest Path | O(ElogV)     | set, heap                    |
| Prim's MST               | O(V$^2$logV) | set, heap                    |
| Kruskal's MST            | O(ElogE)     | array, tree                  |
| Floyd Warshall           | O(n$^3$)     | matrix                       |

&nbsp;

## DFS
```python
graph = ...
visited = set()

def dfs_recursive(n):
    visited.add(n)
    ###### ACTION HERE ######
    ###### ACTION ENDS ######
    for neighbor in graph[n]:
        if neighbor not in visited:
            dfs(neighbor)

def dfs_iterative():
    for key in graph:                   # make sure every node gets a chance (some nodes aren't connected)
        if key in visited: continue     # prevent cycle
        s = collections.deque([key])
        while s:
            n = s.pop()
            visited.add(n)
            ###### ACTION HERE ######
            ###### ACTION ENDS ######
            for neighbor in graph[n]:
                if neighbor not in visited:
                    s.append(neighbor)
```

&nbsp;

## BFS
```python
graph = ...
visited = set()

def bfs():
    for key in graph:                   # make sure every node gets a chance (some nodes aren't connected)
        if key in visited: continue     # prevent cycle
        q = collections.deque([key])
        while q:
            n = q.popleft()
            visited.add(key)
            ###### ACTION HERE ######
            ###### ACTION ENDS ######
            for neighbor in graph[n]:
                if neighbor not in visited:
                    q.append(neighbor)
```

&nbsp;

## Union-Find

Usage: connected components in undirected graphs

Tips: Don't forget to UPDATE PARENTS.

```python
parent = [i for i in range(n)]
rank = [1 for _ in range(n)]

def find(x):
    """
    If not root, keep finding the root of the curr parent and set it as the new parent.
    """
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(x,y):
    """
    If same parent, then they are in the same set already.
    If diff parents,
        higher rank will be the parent.
        if same rank, add one below the other.
    """
    rx,ry = find(x),find(y)
    if rx == ry: return
    if rank[rx] > rank[ry]:
        parent[ry] = rx
    else:
        parent[rx] = ry
        if rank[rx] == rank[ry]:
            rank[ry] += 1

# update parents once again
parent = [find(i) for i in range(n)]
```

&nbsp;

## Topological Sort

Usage: directed acyclic graphs

Tips:
1. Init **graph** & **indegree**.
2. Init **queue** with 0-indegree nodes.
3. Loop q by updating & appending neighbors of 0-indegree.

```python
def topologicalSort(graph):
    n = len(graph)
    indegree = [0]*n
    q = collections.deque()
    visited = set()

    # if graph is not usable, reformat it
    ...

    # compute indegree
    for _,vs in graph.items():
        for v in vs:
            indegree[v] += 1
    
    # get 0-indegree nodes
    for i in range(n):
        if indegree[i] == 0:
            q.append(i)

    # loop queue
    while q:
        node = q.popleft()
        ###### ACTION HERE ######
        ###### ACTION ENDS ######
        visited.add(node)
        for neighbor in graph[n]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                q.append(neighbor)
    
    return
```

&nbsp;

## Dijkstra's Shortest Path
Usage: find shortest path from one node to all other nodes

```python
def dijkstra(graph,start,end):
    heap = [(0,start)]
    visited = set()

    # true Dijkstra
    while heap:
        (p,n) = heapq.heappop(heap)         # get node & path
        if n == end: return p               # return path when found
        if n in visited: continue           # prevent cycle
        visited.add(n)                      # prevent cycle
        for v,w in graph[n]:                # check neighbor
            heapq.heappush(heap,(p+w,v))
    return -1
```

&nbsp;

## Floyd Warshall
Usage: find shortest path from all nodes to all other nodes

```python
def floyd_warshall(graph):
    # init
    N = len(graph)
    dist = [[float("inf")]*N for _ in range(N)]
    
    # set it identical as graph
    for i in range(N):
        for j in range(N):
            dist[i][j] = graph[i][j] # assume graph is 2d matrix
    
    # true Floyd Warshall
    for k in range(N):          # intermediate node
        for i in range(N):      # start node
            for j in range(N):  # end node
                dist[i][j] = min(dist[i][j], dist[i][k]+dist[k][j])  # the triangle rule
    
    return dist
```

&nbsp;

## Prim's MST
Usage: find MST with given node

```python
def primsMST(graph):
    ans = 0
    visited = set()
    heap = [(0,0)] # (edge,node)

    # true Prim
    while len(visited) < len(graph):                # #edges should not exceed #nodes
        e,n = heapq.heappop(heap)                   # get node & edge
        if n in visited: continue                   # prevent cycle
        ans += e                                    # add edge to ans
        visited.add(node)                           # prevent cycle
        for new_e,new_n in graph[n]:                # check neighbor
            if new_n not in visited:
                heapq.heappush(heap, [new_e,new_n])
    return ans
```

&nbsp;

## Kruskal's MST
Usage: find MST with nothing

```python
def KruskalMST(graph):
    MST = []
    parent = [i for i in range(n)]
    rank = [1 for _ in range(n)]
    i = 0   # index for sorted edges in graph
    
    # sort graph by edge weight
    graph = sorted(graph, key=lambda x:x[1])
  
    # true Kruskal
    while len(MST) < len(graph)-1:  # #edges should not exceed #nodes
        u,w,v = graph[i]            # get nodes & edge
        x,y = find(u),find(v)       # get parents
        if x != y:                  # prevent cycle
            MST.append([u,w,v])     # append to MST
            union(x,y)              # now they are connected
        i += 1                      # on to the next edge in graph
```