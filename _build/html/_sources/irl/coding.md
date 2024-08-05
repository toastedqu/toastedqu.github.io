---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---
# Coding interview
A coding interview consists of the following steps:

## Problem
ALWAYS understand the problem first!
- What are some naive examples?
- What data structure(s) are used in this problem?
    - Is it just a 1D array/string?
    - Does it specify the data structure (e.g., Tree)?
    - Does the problem involve relations between entities (Graph)?
    - ...
- What algorithm pattern(s) can be applied to this problem?
    - Are we trying to find something in a sequence (Binary Search)?
    - Is the problem recursive (DFS & BFS)?
    - ...
- By the end of this stage, you should have a rough idea of how to approach this problem.
    - If you don't, you are dead. Go home. Grind leetcode.

ASK clarification questions!
- Confirm the **TYPE & ROLE & DIMENSION** of EACH element in EACH **input & output**.
- Is the input guaranteed to be **VALID**?
- Is the input **SORTED**?
- Is the input **UNIQUE**?
- What's the input **RANGE**?
- Is the output a **full structure** or just some **counter**?
- Construct **edge cases**, which can be used as test cases later.
- Draw a **PAIR OF EXAMPLES** (positive & negative) to the interviewer to confirm your understanding.
    - These examples shall be **specific, sufficiently large, mediocre**.

## Discussion
ALWAYS **COMMUNICATE** with your interviewer!
- State **Brute-Force** first. Talk about its time & space.
- Do NOT code YET.
- Optimize BUDs in naive solution: **Bottlenecks, Unnecessary work, Duplicated work**.
- Look for any **unused info**. Use it!
- Do NOT code YET.
- **Walk through** your entire approach in detail with the interviewer.

When stuck,
- Use **HashMap**!
- Try to solve for **base cases**.
- Try to solve **subproblems**.
- Try other DS/As.

## Data Structure
Identify which data structures are potentially useful.
- Storage:
    - **Array**: store index-specific values.
    - **HashSet**: store unique values, ignore duplicates, reduce search time to $ O(1) $.
    - **HashMap**: store mapping of 2 different entities (e.g., `defaultdict(list), defaultdict(set), Counter()`)
- Process:
    - **Stack**: process things backward (DFS).
    - **Queue**: process things forward (BFS).
    - **Heap**: store & sort, find min/max at each step.
- Question-specific:
    - **LinkedList**: (I personally suck at it. FML if I get asked about this.)
    - **Tree**: DFS & BFS, sometimes Trie.
    - **Graph**: involve relationships between entities.

The single most powerful trick for any non-specific question is:

<center>Use <strong>HashMap</strong> when in doubt.</center>

## Implement & Test
- Walk through your code on your own.
- Modify unusual parts.
- Check null nodes.
- Test small mediocre cases.
- Test edge cases.
- Test special cases.
- Fix bugs whenever necessary.
- Be ready to discuss time & space.

# DS/A
## Array
### Two Pointer
#### Left & Right
**Where**: bi-directional, often on 1d array/string.

**How**:
1. left & right pointers (either facing from both ends, or expanding from middle).
2. Loop. Break when meet.

**Tips**:
- Sort when necessary!
- Use WHILE loop to go over elems that do not break condition.
- 3-pointer (or more) is always an option.

| Question | Solution |
|:---------|:---------|
| [845. Longest Mountain in Array](https://leetcode.com/problems/longest-mountain-in-array/description/) | **Hint: FOR loop; expand both ends.** |
| [5. Longest Palindromic Substring](https://leetcode.com/problems/longest-palindromic-substring/description/) | **Hint: FOR loop; expand one end first, then expand both ends.** |
| [189. Rotate Array](https://leetcode.com/problems/rotate-array/description/) | **Hint: flip array, then flip both subarrays.** |
| [11. Container With Most Water](https://leetcode.com/problems/container-with-most-water/description/) | **Hint: only higher heights can lead to higher volume.**<br>1. WHILE loop between l & r.<br>2. At each step, check if volume needs to be updated.<br>3. Move the pointer with a smaller height.|
| [75. Sort Colors](https://leetcode.com/problems/sort-colors/description/) | **Hint: red & blue pointers for reference, white pointer for iteration.**<br>1. r,b to left,right; w for looping from left.<br>2. If w is 0, swap with r and move both.<br>3. If w is 2, swap with b and move only b.<br>4. If w is 1, move w.|

#### Slow & Fast (Sliding Window)
**Where**: uni-directional, 1d subarray/substring problem

**How**:
1. Init cache(s) with int/arr/map.
2. `fast` for iteration, `slow` for operation (threshold, comparison, etc.)
3. MIN:
    1. Expand `fast` until condition is met.
    2. Shrink `slow` until condition is broken.
4. MAX: 
    1. Expand `fast` until condition is broken.
    2. Dynamically update cache to track condition. If condition is met again, stop `slow`.
```python
def sliding_window(args):
    ## Step 1: init
    s = 0                               ## slow pointer
    cache = collections.Counter(nums)   ## cache

    ## Step 2: iterate
    for f in range(len(nums)):          ## fast pointer
        ## Step 2.1 expand
        cache[f] -= 1

        ## Step 2.2 shrink
        while condition(f):             ## condition state
            cache[s] += 1
            s += 1

    ## Step 3: return whatever the question asks for
    return cache[-1]
```

**Tips**:
- Plan out what to do at every step in every case. It's OK to write them all out at the beginning and optimize afterwards. The key is to think carefully.
- Don't overthink over cases that do not matter. "collections" contains all keys by default, which comes in handy in some cases.

| Question | Solution |
|:---------|:---------|
| [3. Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/description/) | 1. Expand till f char is already visited.<br>2. Remove visited s chars till f char is no longer visited.<br>Cache: visited (set), max len (int). |
| [209. Minimum Size Subarray Sum](https://leetcode.com/problems/minimum-size-subarray-sum/description/) | 1. Expand till curr sum exceeds target.<br>2. Shrink s till curr sum goes below target again.<br>Cache: curr sum (int), min len (int). |
| [424. Longest Repeating Character Replacement](https://leetcode.com/problems/longest-repeating-character-replacement/description/) | 1. Expand till curr len >= max char freq + k. Keep track of char counts, max freq, curr len, and max len (i.e., the answer).<br>2. Shrink s till curr len is smaller again. Update char counts and max freq if necessary.<br>Cache: count (dict), max freq (int), curr len (int), max len (int). |
| [713. Subarray Product Less Than K](https://leetcode.com/problems/subarray-product-less-than-k/description/) | 1. Expand & update curr product till it exceeds k.<br>2. Shrink s till curr product goes below k again.<br>At each step, update #subarrays with the window length.<br>Cache: curr product (int), #subarrays (int). |
| [2302. Count Subarrays With Score Less Than K](https://leetcode.com/problems/count-subarrays-with-score-less-than-k/description/) | 1. Expand till score exceeds k.<br>2. Shrink s till score goes below k again.<br>At each step, update #subarrays with the window length (window length = #subarrays for the current f pointer.)<br>Cache: curr sum (int), #subarrays (int). |
| [1493. Longest Subarray of 1's After Deleting One Element](https://leetcode.com/problems/longest-subarray-of-1s-after-deleting-one-element/description/) | 1. Expand till 0 is met.<br>2. If first time, update deleted. If not, move s after 0 index. Update 0 index to f in both cases.<br>Cache: deleted (bool), 0 index (int), max len (int). |
| [904. Fruit Into Baskets](https://leetcode.com/problems/fruit-into-baskets/description/) | 1. Expand till >=2 types of fruits in basket.<br>2. Move both f & s till only 2 types of fruits in basket. Update window count at each step. Delete the fruit with 0 count.<br>3. The max length is the window length.<br>Cache: basket (dict). |
| [438. Find All Anagrams in a String](https://leetcode.com/problems/find-all-anagrams-in-a-string/description/) | 0. Get p count.<br>1. Expand till length of p first.<br>2. Move both f & s. Update window count at each step. Whenever window count is identical to p count, append s to answer.<br>Cache: p count (dict), window count (dict). |
| [76. Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/description/) | 0. Get t count. Init #missing as len(t).<br>1. Expand till #missing is 0 (i.e., curr window has the substring).<br>2. Shrink till #missing exceeds 0 again.<br>At each step, we update t count by subtracting f char count (and adding slow char count). We should only update #missing when the counts of t chars (which should always be >=0) change.<br>The trick is, the counts of s chars that are not in t will always be below 0, so they will never affect #missing.<br>Cache: t count (dict), #missing chars (int), min substring indices (int) |



### Binary Search
Credits to [zhijun_liao](https://leetcode.com/problems/find-k-th-smallest-pair-distance/solutions/769705/python-clear-explanation-powerful-ultimate-binary-search-template-solved-many-problems/) and [user8301z](https://leetcode.com/problems/find-peak-element/solutions/788474/general-binary-search-thought-process-4-templates/) for helping me understand binary search a lot better.

**Where**: Find a function that maps elements in the left/right half to True and the other to False.
- Although it's mostly applicable to sorted arrays, this is NOT a necessary requirement.

**How**: There are 2 cases to consider. (originally 4, but I think 2 are sufficient for explanation.)
```python
## Find First True (i.e., assume `lefts=False, rights=True`)
while l < r:
    if condition: r = mid   ## If True, ans is on the left (inclusive), so we go left.
    else: l = mid+1         ## If False, ans is on the right, so we go right.

## Find Last True (i.e., assume `lefts=True, rights=False`)
while l < r:
    if condition: l = mid   ## If True, answer is on the right (inclusive), so we go right.
    else: r = mid-1         ## If not, answer is on the left, so we go left.
```
These 2 templates are easily interchangeable by swapping the condition, so we end up with 1 universal template:
```python
def binary_search(nums) -> int:
    def condition(mid) -> bool:
        pass

    l,r = 0,len(nums)-1     ## NOTE: Pay attention to edge cases. Sometimes we need to change this boundary.
    while l < r:
        mid = l+(r-l)//2
        if condition(mid): r = mid
        else: l = mid+1
    return l
```

| Question | Solution |
|:---------|:---------|
| [162. Find Peak Element](https://leetcode.com/problems/find-peak-element/description/) | **Condition: find the first element > its next element.** (`nums[mid] > nums[mid+1]`) |
| [852. Peak Index in a Mountain Array](https://leetcode.com/problems/peak-index-in-a-mountain-array/description/) | **Condition: find the first element > its next element.** (`nums[mid] > nums[mid+1]`) |
| [153. Find Minimum in Rotated Sorted Array](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/description/) | **Condition: find the first element <= everything on its right.** (`nums[mid] <= nums[r]`) |
| [528. Random Pick with Weight](https://leetcode.com/problems/random-pick-with-weight/description/) | **Condition: find the first index with a cumulative probability >= a random probability.** (`self.w[mid] >= p`)<br>NOTE: convert input array to cumulative probabilities. |
| [34. Find First and Last Position of Element in Sorted Array](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/description/) | **Condition: find the first element >= input number.** (`nums[mid] >= n`)<br>1. Write this condition in a func.<br>2. The first occurrence is bs(target). (i.e., first element >= target)<br>3. The last occurrence is bs(target+1)-1. (i.e., first element >= target+1, then on its left)<br>4. If first <= last, ans is valid. Else, target not in array.<br>NOTE: given the way `last` is defined, `r=len(nums)`. |
| [362. Design Hit Counter](https://leetcode.com/problems/design-hit-counter/description/) | **Condition: find the index where `timestamp-300` would have been inserted at.** (`self.hits[mid] > target`)<br>1. Write this condition in a func.<br>2. #hits = `len(self.hits)-self.bs(timestamp-300)`. |
| [540. Single Element in a Sorted Array](https://leetcode.com/problems/single-element-in-a-sorted-array/description/) | **Condition: find the first element that distorts the duplicate order.**<br>- For the input array, if `i` is odd / `i-1` is even, the condition should always hold: `nums[i] == nums[i-1]`.<br>- However, the element we are looking for violates this condition, so we have 2 cases:<br>- if `mid` is odd and `nums[mid] != nums[mid-1]`.<br>- if `mid` is even and `nums[mid] != nums[mid+1]`. |
| [658. Find K Closest Elements](https://leetcode.com/problems/find-k-closest-elements/description/) | **Condition: find the first element that starts the subarray.**<br>- Since `a` is closer than `b` if `\|a-x\| <= \|b-x\|` and `a < b`, it is straightforward that our first element in the subarray satisfies `\|nums[mid]-x\| <= \|nums[mid+k]-x\|`.<br>- It's the same as `x-nums[mid] <= nums[mid+k]-x`. |
| [719. Find K-th Smallest Pair Distance](https://leetcode.com/problems/find-k-th-smallest-pair-distance/description/) | **Condition: find the smallest distance that has >=k pairs within its range.**<br>1. Write a function that uses two pointers (slow & fast) to find the #pairs with distances <= input distance. Return True if there are >=k pairs, else False.<br>2. Sort `nums` for two pointers to work.<br>3. Set `r=nums[-1]-nums[0]` because our search space is not `nums` but distances.<br>4. Search. |



### Merge Intervals
**Where**: interval problems

**How**: depends on the question tbh.

<center>
<img src="/images/dsa/merge_interval.jpg" width=50%/>
</center>

**Tips**:
- There are only 2 conditions for 2 intervals to overlap: **front.start <= back.end** AND **back.start <= front.end**.
- Use MIN/MAX to update merged `start`/`end`.
- Sort by `start`/`end` when necessary.

| Question | Solution |
|:---------|:---------|
| [56. Merge Intervals](https://leetcode.com/problems/merge-intervals/description/) | 0. Sort.<br>1. Loop.<br>Append interval till `front.end >= back.start`.<br>Update `front.end` with max end. |
| [759. Employee Free Time (opposite of 56)](https://leetcode.com/problems/employee-free-time/description/) | 0. Flatten & Sort a workhour array by start time.<br>1. Loop.<br>Append interval till `front.end >= back.start`.<br>Update `front.end` with max end.<br>2. Loop through the array of merged workhours. The intervals in between (i.e., `[front.end, back.start]`) are free hours. |
| [57. Insert Interval](https://leetcode.com/problems/insert-interval/description/) | 0. Init left & right subarrays.<br>1. Loop.<br>Append interval to left if `interval.end < newinterval.start` (i.e., on the left of new interval)<br>Append interval to right if `interval.start > newinterval.end` (i.e., on the right of new interval).<br>Else, we find the insertion position. Keep track of the position with MIN of start and MAX of end.<br>2. Concatenate left, `[insert_start, insert_end]`, and right together as the answer.|
| [986. Interval List Intersections](https://leetcode.com/problems/interval-list-intersections/description/) | 0. Init a pointer for each list.<br>1. Loop till length.<br>If intersect, append `[max_start, min_end]` to answer (i.e., the intersection interval)<br>Else, move the pointer with the smaller end value. |



### Dynamic Programming
**Where**: The OG problem can be divided into smaller overlapping subproblems with an optimal substructure.

**How**:
1. Choose a method: Decide between the two main methods of DP: Top-Down (Memoization) and Bottom-Up (Tabulation).
    - Top-Down (**Memoization**): Break problem into smaller subproblems. Store results for each subproblem in an array/hash.
    - Bottom-Up (**Tabulation**): Start from the simplest subproblems and iteratively solve larger problems. Store results in a table (i.e., 2d array).
2. Define the state:
    - Variables: Determine what parameters can uniquely identify a subproblem, which will be used to index into your memoization table or array.
    - Transition: Define how to break your problem into subproblems.
3. Initialize (DP table & Base cases) & Iterate (based on the defined state transition).

```python
def DP(args):
    ## Step 1: Init
    dp = [0]*(n+1)          ## DP table
    dp[0] = 1               ## base case

    ## Step 2: Iterate
    for i in range(1, n+1):
        dp[i] = dp[i-1]+1   ## state transition

    ## Step 3: Return end case
    return dp[n]
```
| Question | Solution |
|:---------|:---------|
| [39. Combination Sum](https://leetcode.com/problems/combination-sum/description/) | **DP: store the combinations to get the desired amount with the given coins.**<br>Base: f(0)=[]; f(coin)=[coin].<br>Transition: loop through each candidate and all numbers between candidate and target.<br>- If the number is the candidate, store the candidate.<br>- Else, append candidate to all combinations of the previous store. (`dp[i] = dp[i-c]+[c]`) |
| [53. Maximum Subarray](https://leetcode.com/problems/maximum-subarray/description/) | **DP: store max attainable sum for the given index.**<br>Base: all 0.<br>Transition: loop through each index.<br>- If previous max sum is bigger than 0, curr max sum = prev + curr num.<br>- Else, curr max sum = curr num (because adding negative sum is not max.)<br>Return `max(dp)`.|
| [62. Unique Paths](https://leetcode.com/problems/unique-paths/description/) | **DP: store #unique paths for the given point.**<br>Base: set all starting edges to 1.<br>Transition: loop through each point from top left to bottom right.<br>- `dp[i][j] = dp[i-1][j] + dp[i][j-1]` |
| [139. Word Break](https://leetcode.com/problems/word-break/description/) | **DP: store whether there is a word in worddict that ends with index i in string s.**<br>Base: all False.<br>Transition: loop through each index and each word in dict.<br>- If there is such a word and (the preceding word also exists or this word is the first word), then `dp[i]=True`. |
| [322. Coin Change](https://leetcode.com/problems/coin-change/description/) | **DP: store min #coins for each amount till target.**<br>Base: f(0)=0; f(coin)=1.<br>Transition: loop through each amount and each coin.<br>- If curr ammount is bigger than the coin, `dp[i] = min(dp[i-coin] for coin in coins)+1`.<br>- Else, `dp[i]=float("inf")`. |
| [377. Combination Sum IV](https://leetcode.com/problems/combination-sum-iv/description/) | **DP: store #combinations for each target till target.**<br>Base: f(0)=1.<br>Transition: loop through each target and each number.<br>- `dp[i] = sum(dp[i-num] for num in nums)` |
| [1143. Longest Common Subsequence](https://leetcode.com/problems/longest-common-subsequence/description/) | **DP: store max common subsequence length for each index in each string.**<br>Base: all 0.<br>Transition: loop through each index in each string.<br>- If two chars are the same, `dp[i+i][j+1] = dp[i][j]+1`<br>- Else, `dp[i+1][j+1] = max(dp[i][j+1],dp[i+1][j])` |
| [416. Partition Equal Subset Sum](https://leetcode.com/problems/partition-equal-subset-sum/description/) | **2 DPs: both sets store the sums of all subsets till this index. 1 for main store, 1 for looping.**<br>Base: f(0)=0.<br>Transition: loop through each index and each stored sum for previous index.<br>1. Init looping dp. Also, init target sum (i.e., half of total sum).<br>2. For each stored sum for previous index, stored in the main dp,<br>- At each number, we have 2 options: add it, or don't.<br>- When add it, we append `sum+num` to the looping dp.<br>- When don't, we append `sum` to the looping dp.<br>- Do both options at each loop because there is no if-else for this problem.<br>3. Update main dp with looping dp. If target sum in main dp, return True. If no true after all loops, return False. |
| [494. Target Sum](https://leetcode.com/problems/target-sum/description/) | **DP: store mappings of "(index, target) -> #ways to get to target from curr index".**<br>**DFS: return #ways to get to target from curr index.**<br>Leaf case: #ways is either 1 (if hit target) or 0 (if not).<br>Break case: if we already visited (index, target), no need to loop further.<br>Loop: #ways at curr index = #ways at left + #ways at right.<br>Start with (0,0). |
| [329. Longest Increasing Path in a Matrix](https://leetcode.com/problems/longest-increasing-path-in-a-matrix/description/) | **DP: store max attainable len for each point.**<br>**DFS: return max attainable len for curr point.**<br>Break case: return 0 if out of boundary or not increasing (keep track of previous number to check if increasing).<br>Loop: if curr point not visited, set curr point store to max of all 4 movements plus 1.<br>Start with all points on matrix. Return the max len. |



## Linked List
My peanut brain literally cannot understand linked list problems, so take this section with a grain of salt.

**Where**: linked list

**Tips**: 
- Problem: search, insert, delete, reverse, etc.
- Diagram: if you can, draw them.
- Pointer: carefully keep track of what your "prev" and "curr" pointers are doing at every single step. Are they doing add, remove, or rearrange nodes?
    - It involves 3 nodes instead of 2 to make a full reverse.
- Dummies: don't be afraid to init bunch of temp nodes. They are all O(1) anyway.
- Edge cases: empty list, single node list, head/tail nodes, etc. 

| Question | Solution |
|:---------|:---------|
| [19. Remove Nth Node From End of List](https://leetcode.com/problems/remove-nth-node-from-end-of-list/description/) | 1. Two pointers at head.<br>2. Loop fast first till `n`. If fast cannot reach `n`, no need to remove.<br>3. Loop both fast and slow. When fast finishes, slow will be at `n-1`th node.<br>4. Put slow.next to slow.next.next. Return head. |
| [23. Merge k Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/description/) | 1. Init a dummy node for the later merged list. Init a pointer for the list.<br>2. Init a heap with first nodes of each list, storing both value and index for the node.<br>3. Recursively pop & push values and indices to the heap till we run out of nodes. |
| [24. Swap Nodes in Pairs](https://leetcode.com/problems/swap-nodes-in-pairs/description/) | 1. Init a dummy node for the head. Init 2 pointers (`prev` & `curr`) to `dummy` & `head`.<br>2. Keep track of 3 nodes in each iteration of the WHILE loop on `curr` & `post`.<br>2.1. Point curr & post to new curr & new post.<br>2.2. Point `curr.next` to `post.next`.<br>2.3. Point `post.next` to `curr`.<br>2.4. Point `prev.next` to `post`.<br>2.5. Move `prev` to `curr`. |

<!-- ```python
def reverseList(self, head):
    prev, curr = None, head
    while curr:
        temp = curr.next
        curr.next = prev
        prev, curr = curr, temp
    return prev
``` -->



## Stack & Queue
### Monotonic Stack
**Where**: increasing/decreasing trend

**Tips**:
- Understand clearly what the variable for comparison is. Use stack to store
    - the variable for comparison
    - another variable closely associated with it
- If an element violates the condition, enter a WHILE loop. Continuously update and pop values from stack until
    - this element stays in condition.
    - the stack is EMPTY.
- Else, append element to stack.

```python
## 739. Daily Temperatures
def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
    ans = [0]*len(temperatures)
    s = []  ## store indices of monotonically decreasing temps
    for i,t in enumerate(temperatures):
        while s and temperatures[s[-1]] < t:  ## if a higher temp is met
            ans[s[-1]] = i-s[-1]              ## update ans
            s.pop()                           ## pop till no higher temp
        s.append(i)                           ## append unanswered index to stack
    return ans
```



## Tree
### DFS
**Where**: longest/specific search problems

**How?**
```python
def dfs_pre(node):
    if is_end_case(): return True
    if break_condition(): return False
    ####### ACTION HERE ######
    ####### ACTION ENDS ######
    dfs_pre(node.left)
    dfs_pre(node.right)

def dfs_in(node):
    if is_end_case(): return True
    if break_condition(): return False
    dfs_in(node.left)
    ####### ACTION HERE ######
    ####### ACTION ENDS ######
    dfs_in(node.right)

def dfs_post(node):
    if is_end_case(): return True
    if break_condition(): return False
    dfs_post(node.left)
    dfs_post(node.right)
    ####### ACTION HERE ######
    ####### ACTION ENDS ######
```

**Tips**: 
- Choose carefully what you want: pre-order / in-order / post-order.
- Each loop should ONLY focus on
    1) the end/base case
    2) the curr node
- When returning bool, specify both True and False end/base cases.



### BFS
**Where**: shortest search problems

**Tips**: queue/priority queue

```python
def bfs(node):
    if not node: return
    q = collections.deque([node])
    while q:
        node = q.popleft()
        ####### ACTION HERE ######
        ####### ACTION ENDS ######
        if node.left:  q.append(node.left)
        if node.right: q.append(node.right)
```



## Heap
**Where**: get min/max fast
| Action | Time |
|------|----|
| top()  | O(1) |
| insert()  | O(logn) |
| remove()  | O(logn) |
| heapify() | O(n) |

### Two Heap
**Where**: scheduling, median, any problem that involves both min and max somehow.

**Tips**:
- Set up 2 heaps:
    - small: max heap (i.e., negative min heap)
    - large: min heap
- Use their length as storage condition
- Do NOT pop when looking up items. Use index (0 for root).

```python
## 295. Find Median from Data Stream
class MedianFinder:
    def __init__(self):
        self.small = [] ## heap for the smaller half (negative so that min heap works)
        self.large = [] ## heap for the larger half

    def addNum(self, num: int) -> None: ## O(logn)
        ## It doesn't really matter which one has one more value than the other. But be consistent during interview.
        ## In this case, we allow "small" to store one more value than "large" when #nums is odd.
        if len(self.small)==len(self.large):                                 ## if #nums is now even
            heapq.heappush(self.small, -heapq.heappushpop(self.large, num))  ## push new num to "large", pop the smallest from "large", put it in "small"
        else:
            heapq.heappush(self.large, -heapq.heappushpop(self.small, -num)) ## push new num to "small", pop the largest from "small", put it in "large"

    def findMedian(self) -> float:      ## O(1)
        if len(self.small)==len(self.large):        
            return (self.large[0]-self.small[0])/2
        else:
            return -self.small[0]
```



## Graph
| Algorithm                | Time         | DS needed                    |
|--------------------------|--------------|------------------------------|
| DFS                      | O(n)         | set (and stack if iteration) |
| BFS                      | O(n)         | set, queue                   |
| Union-Find               | O(nlogn)     | array, tree                  |
| Topological Sort         | O(n)         | array, set, queue            |
| Dijkstra's Shortest Path | O(ElogV)     | set, heap                    |
| Prim's MST               | O(V$ ^2 $logV) | set, heap                    |
| Kruskal's MST            | O(ElogE)     | array, tree                  |
| Floyd Warshall           | O(n$ ^3 $)     | matrix                       |

### DFS
```python
graph = ...
visited = set()

def dfs_recursive(n):
    visited.add(n)
    ####### ACTION HERE ######
    ####### ACTION ENDS ######
    for neighbor in graph[n]:
        if neighbor not in visited:
            dfs(neighbor)

def dfs_iterative():
    for key in graph:                   ## make sure every node gets a chance (some nodes aren't connected)
        if key in visited: continue     ## prevent cycle
        s = collections.deque([key])
        while s:
            n = s.pop()
            visited.add(n)
            ####### ACTION HERE ######
            ####### ACTION ENDS ######
            for neighbor in graph[n]:
                if neighbor not in visited:
                    s.append(neighbor)
```



### BFS
```python
graph = ...
visited = set()

def bfs():
    for key in graph:                   ## make sure every node gets a chance (some nodes aren't connected)
        if key in visited: continue     ## prevent cycle
        q = collections.deque([key])
        while q:
            n = q.popleft()
            visited.add(key)
            ####### ACTION HERE ######
            ####### ACTION ENDS ######
            for neighbor in graph[n]:
                if neighbor not in visited:
                    q.append(neighbor)
```



### Union-Find

**Where**: connected components in undirected graphs

**Tips**: Don't forget to UPDATE PARENTS.

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

## update parents once again
parent = [find(i) for i in range(n)]
```



### Topological Sort

**Where**: directed acyclic graphs

**Tips**:
1. Init **graph** & **indegree**.
2. Init **queue** with 0-indegree nodes.
3. Loop q by updating & appending neighbors of 0-indegree.

```python
def topologicalSort(graph):
    n = len(graph)
    indegree = [0]*n
    q = collections.deque()
    visited = set()

    ## if graph is not usable, reformat it
    ...

    ## compute indegree
    for _,vs in graph.items():
        for v in vs:
            indegree[v] += 1
    
    ## get 0-indegree nodes
    for i in range(n):
        if indegree[i] == 0:
            q.append(i)

    ## loop queue
    while q:
        node = q.popleft()
        ####### ACTION HERE ######
        ####### ACTION ENDS ######
        visited.add(node)
        for neighbor in graph[n]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                q.append(neighbor)
    
    return
```



### Dijkstra's Shortest Path
**Where**: find shortest path from one node to all other nodes

```python
def dijkstra(graph,start,end):
    heap = [(0,start)]
    visited = set()

    ## true Dijkstra
    while heap:
        (p,n) = heapq.heappop(heap)         ## get node & path
        if n == end: return p               ## return path when found
        if n in visited: continue           ## prevent cycle
        visited.add(n)                      ## prevent cycle
        for v,w in graph[n]:                ## check neighbor
            heapq.heappush(heap,(p+w,v))
    return -1
```



### Floyd Warshall
**Where**: find shortest path from all nodes to all other nodes

```python
def floyd_warshall(graph):
    ## init
    N = len(graph)
    dist = [[float("inf")]*N for _ in range(N)]
    
    ## set it identical as graph
    for i in range(N):
        for j in range(N):
            dist[i][j] = graph[i][j] ## assume graph is 2d matrix
    
    ## true Floyd Warshall
    for k in range(N):          ## intermediate node
        for i in range(N):      ## start node
            for j in range(N):  ## end node
                dist[i][j] = min(dist[i][j], dist[i][k]+dist[k][j])  ## the triangle rule
    
    return dist
```



### Prim's MST
**Where**: find MST with given node

```python
def primsMST(graph):
    ans = 0
    visited = set()
    heap = [(0,0)] ## (edge,node)

    ## true Prim
    while len(visited) < len(graph):                ## #edges should not exceed #nodes
        e,n = heapq.heappop(heap)                   ## get node & edge
        if n in visited: continue                   ## prevent cycle
        ans += e                                    ## add edge to ans
        visited.add(node)                           ## prevent cycle
        for new_e,new_n in graph[n]:                ## check neighbor
            if new_n not in visited:
                heapq.heappush(heap, [new_e,new_n])
    return ans
```



### Kruskal's MST
**Where**: find MST with nothing

```python
def KruskalMST(graph):
    MST = []
    parent = [i for i in range(n)]
    rank = [1 for _ in range(n)]
    i = 0   ## index for sorted edges in graph
    
    ## sort graph by edge weight
    graph = sorted(graph, key=lambda x:x[1])
  
    ## true Kruskal
    while len(MST) < len(graph)-1:  ## #edges should not exceed #nodes
        u,w,v = graph[i]            ## get nodes & edge
        x,y = find(u),find(v)       ## get parents
        if x != y:                  ## prevent cycle
            MST.append([u,w,v])     ## append to MST
            union(x,y)              ## now they are connected
        i += 1                      ## on to the next edge in graph
```