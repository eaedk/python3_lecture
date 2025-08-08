# Python DSA & Coding Interview: Full Course (No Libraries)

> **Goal:** master core data structures, algorithms, and interview patterns by **re‑implementing everything from scratch** (no `heapq`, no `sorted()`, no built‑in set/dict tricks when reimplementing), documenting code, and drilling with targeted problems.

---

## How to Use This Course

* **Write first, read second.** Implement each structure/algorithm *before* reading the provided reference solution.
* **No-libraries mode.** Unless explicitly allowed, avoid convenience methods. Rebuild primitives to understand internals.
* **Time it.** Practice with a timer (20–45 min/problem). After solving, explain your approach out loud.
* **Document.** Every function/class has a docstring with purpose, invariants, and complexity.
* **Review loop.** After each module: 3 “must‑know” problems in under 60 minutes.

---

## 30‑Day Learning Plan (2–3h/day)

| Day   | Focus                                                             | Output                          |
| ----- | ----------------------------------------------------------------- | ------------------------------- |
| 1     | Course rules, asymptotics, Python minimalism, testing template    | Big‑O cheat sheet, test harness |
| 2–3   | Arrays & Strings (two pointers, sliding window)                   | 6 problems solved + notes       |
| 4     | Prefix sums, difference arrays, Kadane                            | 4 problems                      |
| 5–6   | Linked Lists (singly/doubly), cycle, reverse, merge               | Implement LL + 6 problems       |
| 7     | Stacks & Queues (array + linked)                                  | Implement both + 4 problems     |
| 8     | Custom Hash Table (chaining + open addressing)                    | Implement HT + 4 problems       |
| 9     | Recursion basics, backtracking patterns                           | 5 problems                      |
| 10–11 | Sorting from scratch (bubble, insertion, selection, merge, quick) | Implement all + 5 compare Qs    |
| 12    | Binary search patterns (standard + on answer)                     | 6 problems                      |
| 13–14 | Trees & BST (traversals, validation, LCA)                         | Implement Tree + 6 problems     |
| 15    | Heaps/Priority Queue from scratch                                 | Implement bin-heap + 4 problems |
| 16–17 | Graphs (adj list, BFS/DFS, topological sort)                      | Implement Graph + 6 problems    |
| 18    | Shortest paths (BFS unweighted, Dijkstra w/ custom PQ)            | 4 problems                      |
| 19–21 | Dynamic Programming I–III (1D/2D, knapsack, LIS)                  | 8 problems                      |
| 22    | Greedy + intervals + scheduling                                   | 5 problems                      |
| 23    | Bit tricks (masks, parity, subsets)                               | 5 problems                      |
| 24    | String algorithms (Rabin–Karp, KMP)                               | Implement + 3 problems          |
| 25    | Mix‑set drills (blend multiple topics)                            | 6 problems                      |
| 26    | Systematic interview strategy + whiteboard checklist              | Personal playbook               |
| 27–28 | 2 mock interviews (algorithms + systems‑thinking)                 | Feedback notes                  |
| 29    | Final review: 15‑problem gauntlet                                 | Score & gaps                    |
| 30    | Capstone interview                                                | Full debrief + next steps       |

> Optional pace: double the days for 1–1.5h/day.

---

## Ground Rules & Testing Harness

```python
"""
Minimal test harness to run assertions without external libs.
Run: python file.py
"""

def assert_eq(actual, expected, msg=""):
    if actual != expected:
        raise AssertionError(f"{msg} Expected {expected}, got {actual}")

class Tests:
    passed = 0
    failed = 0
    
    @classmethod
    def case(cls, name, fn):
        try:
            fn()
            cls.passed += 1
            print(f"[PASS] {name}")
        except Exception as e:
            cls.failed += 1
            print(f"[FAIL] {name}: {e}")

    @classmethod
    def summary(cls):
        print(f"\nPassed: {cls.passed}  Failed: {cls.failed}")

if __name__ == "__main__":
    # Add Tests.case("name", lambda: ...) in modules below.
    Tests.summary()
```

---

## Big‑O Cheat (keep handy)

* Arrays scan: O(n)
* Sorts: mergesort O(n log n) stable; quicksort avg O(n log n), worst O(n²)
* Hash average ops: O(1); worst O(n)
* BST balanced ops: O(log n); unbalanced: O(n)
* Graph BFS/DFS: O(V+E)
* Dijkstra (binary heap): O((V+E) log V)
* DP often: states × transitions

---

# Module 0 — Python Minimalism for Interviews

**Aim:** use basic syntax only. No `sorted()`, no `heapq`, no `collections`.

* Loops, conditionals, list indexing
* Manual swaps: `a[i], a[j] = a[j], a[i]`
* String -> list for in‑place editing
* Custom classes with `__init__`, methods, and docstrings
* Avoid recursion depth issues (<\~1000). Prefer iterative when deep.

---

# Module 1 — Arrays & Strings

### Concepts

* Two Pointers (same/opposite direction)
* Sliding Window (fixed/variable)
* Prefix/Suffix arrays
* In‑place operations & invariants

### From‑scratch Utilities

```python
class DynamicArray:
    """Simplified dynamic array using Python list but manual resize semantics."""
    def __init__(self, capacity=4):
        self._n = 0
        self._cap = capacity
        self._data = [None] * capacity
    def __len__(self): return self._n
    def _resize(self, new_cap):
        new_data = [None] * new_cap
        for i in range(self._n):
            new_data[i] = self._data[i]
        self._data = new_data
        self._cap = new_cap
    def append(self, x):
        if self._n == self._cap:
            self._resize(self._cap * 2)
        self._data[self._n] = x
        self._n += 1
    def get(self, i):
        if i < 0 or i >= self._n:
            raise IndexError("out of bounds")
        return self._data[i]
```

### Patterns & Example

**1. Two Pointers – Remove Duplicates from Sorted Array (in‑place)**

```python
def remove_dups_sorted(a):
    """Return new length after removing duplicates in-place. O(n)."""
    if not a: return 0
    w = 1
    for r in range(1, len(a)):
        if a[r] != a[w-1]:
            a[w] = a[r]
            w += 1
    return w
```

**2. Sliding Window – Longest Substring with ≤ K Distinct**

```python
def longest_k_distinct(s, k):
    """Without built-in dict: implement tiny map via arrays of size 128 (ASCII)."""
    count = [0]*128
    distinct = 0
    best = 0
    l = 0
    for r in range(len(s)):
        c = ord(s[r])
        if count[c] == 0:
            distinct += 1
        count[c] += 1
        while distinct > k:
            cl = ord(s[l])
            count[cl] -= 1
            if count[cl] == 0:
                distinct -= 1
            l += 1
        if r-l+1 > best:
            best = r-l+1
    return best
```

### Drill (pick 4)

* Reverse words in a string (in place with char array)
* Move zeros to end preserving order
* Minimum window substring (ASCII map, expand/contract)
* Kadane maximum subarray sum

**Reference: Kadane**

```python
def max_subarray(a):
    best = a[0]
    cur = a[0]
    for i in range(1, len(a)):
        cur = a[i] if cur < 0 else cur + a[i]
        if cur > best:
            best = cur
    return best
```

---

# Module 2 — Linked Lists

### Structures

```python
class ListNode:
    def __init__(self, val=0, nxt=None):
        self.val = val
        self.next = nxt

class LinkedList:
    def __init__(self):
        self.head = None
    def push_front(self, x):
        self.head = ListNode(x, self.head)
    def to_list(self):
        out, cur = [], self.head
        while cur:
            out.append(cur.val)
            cur = cur.next
        return out
```

### Core Routines

* Reverse list (iterative)
* Detect cycle (Floyd)
* Merge two sorted lists
* Remove Nth from end (two pointers)

```python
def reverse_list(head):
    prev = None
    cur = head
    while cur:
        nxt = cur.next
        cur.next = prev
        prev = cur
        cur = nxt
    return prev
```

---

# Module 3 — Stacks & Queues

### Array Stack & Queue

```python
class Stack:
    def __init__(self): self.a=[]
    def push(self,x): self.a.append(x)
    def pop(self): return self.a.pop() if self.a else None
    def peek(self): return self.a[-1] if self.a else None
    def empty(self): return len(self.a)==0

class Queue:
    def __init__(self): self.in_a=[]; self.out_a=[]
    def push(self,x): self.in_a.append(x)
    def _pour(self):
        if not self.out_a:
            while self.in_a:
                self.out_a.append(self.in_a.pop())
    def pop(self):
        self._pour()
        return self.out_a.pop() if self.out_a else None
    def peek(self):
        self._pour()
        return self.out_a[-1] if self.out_a else None
```

**Problems:** valid parentheses (stack), daily temperatures (stack of indices), sliding window max (deque imitation with list).

---

# Module 4 — Hash Table from Scratch

### Separate Chaining

```python
class Entry:
    def __init__(self, k, v, nxt=None):
        self.k, self.v, self.next = k, v, nxt

class HashTable:
    """String key hash table (ASCII), chaining."""
    def __init__(self, cap=8):
        self.n=0; self.cap=cap; self.b=[None]*cap
    def _h(self, s):
        h=0
        for ch in s:
            h = (h*131 + ord(ch)) & 0x7fffffff
        return h % self.cap
    def _resize(self):
        old = self.b
        self.cap *= 2
        self.b = [None]*self.cap
        self.n = 0
        for head in old:
            cur=head
            while cur:
                self.put(cur.k, cur.v)
                cur = cur.next
    def put(self, k, v):
        if self.n*10 >= self.cap*7: # load > 0.7
            self._resize()
        i = self._h(k)
        cur = self.b[i]
        while cur:
            if cur.k==k:
                cur.v=v; return
            cur=cur.next
        self.b[i]=Entry(k,v,self.b[i])
        self.n+=1
    def get(self, k):
        i=self._h(k); cur=self.b[i]
        while cur:
            if cur.k==k: return cur.v
            cur=cur.next
        return None
    def remove(self, k):
        i=self._h(k); cur=self.b[i]; prev=None
        while cur:
            if cur.k==k:
                if prev: prev.next=cur.next
                else: self.b[i]=cur.next
                self.n-=1
                return True
            prev,cur=cur,cur.next
        return False
```

**Drill:** two‑sum using custom HT, first unique char, anagram check without sort.

---

# Module 5 — Recursion & Backtracking

**Patterns:** build, choose, explore, un‑choose.

* Subsets, permutations, combinations
* N‑Queens
* Word search

```python
def subsets(nums):
    res=[]; path=[]
    def dfs(i):
        if i==len(nums):
            res.append(path[:]); return
        dfs(i+1)
        path.append(nums[i])
        dfs(i+1)
        path.pop()
    dfs(0)
    return res
```

---

# Module 6 — Sorting From Scratch

```python
def bubble(a):
    n=len(a)
    for i in range(n):
        swapped=False
        for j in range(0,n-1-i):
            if a[j]>a[j+1]:
                a[j],a[j+1]=a[j+1],a[j]
                swapped=True
        if not swapped: break

def insertion(a):
    for i in range(1,len(a)):
        key=a[i]; j=i-1
        while j>=0 and a[j]>key:
            a[j+1]=a[j]; j-=1
        a[j+1]=key

def selection(a):
    n=len(a)
    for i in range(n):
        m=i
        for j in range(i+1,n):
            if a[j]<a[m]: m=j
        a[i],a[m]=a[m],a[i]

def merge_sort(a):
    if len(a)<=1: return a
    mid=len(a)//2
    L=merge_sort(a[:mid]); R=merge_sort(a[mid:])
    i=j=0; out=[]
    while i<len(L) and j<len(R):
        if L[i]<=R[j]: out.append(L[i]); i+=1
        else: out.append(R[j]); j+=1
    while i<len(L): out.append(L[i]); i+=1
    while j<len(R): out.append(R[j]); j+=1
    return out

def quick_sort(a, l=0, r=None):
    if r is None: r=len(a)-1
    if l>=r: return
    p=a[(l+r)//2]
    i, j = l, r
    while i<=j:
        while a[i]<p: i+=1
        while a[j]>p: j-=1
        if i<=j:
            a[i],a[j]=a[j],a[i]
            i+=1; j-=1
    quick_sort(a,l,j); quick_sort(a,i,r)
```

**Why each:** stability, space, typical use.

---

# Module 7 — Binary Search (and on Answer)

```python
def bsearch(a, x):
    l, r = 0, len(a)-1
    while l<=r:
        m=(l+r)//2
        if a[m]==x: return m
        if a[m]<x: l=m+1
        else: r=m-1
    return -1
```

**On answer:** minimize `f(x) >= target` or capacity problems.

```python
def min_capacity(weights, days):
    def can(c):
        used=1; cur=0
        for w in weights:
            if w>c: return False
            if cur+w>c:
                used+=1; cur=0
            cur+=w
        return used<=days
    l, r = 1, sum(weights)
    ans=r
    while l<=r:
        m=(l+r)//2
        if can(m): ans=m; r=m-1
        else: l=m+1
    return ans
```

---

# Module 8 — Trees & BST

```python
class TNode:
    def __init__(self, val, left=None, right=None):
        self.val=val; self.left=left; self.right=right

def inorder(root):
    out=[]; stack=[]; cur=root
    while cur or stack:
        while cur:
            stack.append(cur); cur=cur.left
        cur=stack.pop(); out.append(cur.val)
        cur=cur.right
    return out

def is_bst(root):
    stack=[]; prev=None; cur=root
    while cur or stack:
        while cur:
            stack.append(cur); cur=cur.left
        cur=stack.pop()
        if prev is not None and cur.val<=prev: return False
        prev=cur.val; cur=cur.right
    return True
```

**LCA (BST):** walk down comparing values.

---

# Module 9 — Heaps / Priority Queue (No `heapq`)

```python
class MinHeap:
    def __init__(self): self.a=[]
    def _up(self,i):
        while i>0:
            p=(i-1)//2
            if self.a[p]<=self.a[i]: break
            self.a[p],self.a[i]=self.a[i],self.a[p]
            i=p
    def _down(self,i):
        n=len(self.a)
        while True:
            l=2*i+1; r=2*i+2; s=i
            if l<n and self.a[l]<self.a[s]: s=l
            if r<n and self.a[r]<self.a[s]: s=r
            if s==i: break
            self.a[i],self.a[s]=self.a[s],self.a[i]
            i=s
    def push(self,x):
        self.a.append(x); self._up(len(self.a)-1)
    def pop(self):
        if not self.a: return None
        top=self.a[0]
        last=self.a.pop()
        if self.a:
            self.a[0]=last; self._down(0)
        return top
    def peek(self): return self.a[0] if self.a else None
```

**Use cases:** top‑k, k‑way merge, Dijkstra.

---

# Module 10 — Graphs

```python
class Graph:
    def __init__(self, n):
        self.n=n
        self.adj=[[] for _ in range(n)]
    def add_edge(self,u,v,undirected=False):
        self.adj[u].append(v)
        if undirected:
            self.adj[v].append(u)

def bfs(g, s):
    q=[s]; seen=[False]*g.n; seen[s]=True; order=[]
    i=0
    while i<len(q):
        u=q[i]; i+=1; order.append(u)
        for v in g.adj[u]:
            if not seen[v]:
                seen[v]=True; q.append(v)
    return order

def dfs(g, s):
    seen=[False]*g.n; order=[]
    def go(u):
        seen[u]=True; order.append(u)
        for v in g.adj[u]:
            if not seen[v]: go(v)
    go(s)
    return order
```

**Topological sort:** DFS postorder or Kahn’s (indegrees via array).

**Dijkstra (with custom MinHeap)**

```python
def dijkstra(g, src, weights):
    INF=10**18
    dist=[INF]*g.n; dist[src]=0
    pq=MinHeap(); pq.push((0,src))
    # MinHeap supports tuple compare since first element decides
    while pq.peek() is not None:
        d,u = pq.pop()
        if d!=dist[u]:
            continue
        for idx,v in enumerate(g.adj[u]):
            w = weights[u][idx]
            nd = d + w
            if nd < dist[v]:
                dist[v]=nd
                pq.push((nd,v))
    return dist
```

---

# Module 11 — Dynamic Programming

**Patterns:**

* 1D DP (climb stairs, house robber)
* 2D DP (edit distance, unique paths)
* Knapsack (0/1 and unbounded)
* LIS (O(n log n) with manual binary search)

```python
def edit_distance(a,b):
    n,m=len(a),len(b)
    dp=[[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0]=i
    for j in range(m+1): dp[0][j]=j
    for i in range(1,n+1):
        for j in range(1,m+1):
            cost = 0 if a[i-1]==b[j-1] else 1
            dp[i][j]=min(
                dp[i-1][j]+1,
                dp[i][j-1]+1,
                dp[i-1][j-1]+cost
            )
    return dp[n][m]
```

**0/1 Knapsack**

```python
def knapsack(W, wt, val):
    n=len(wt)
    dp=[[0]*(W+1) for _ in range(n+1)]
    for i in range(1,n+1):
        for w in range(W+1):
            dp[i][w]=dp[i-1][w]
            if wt[i-1]<=w:
                cand = dp[i-1][w-wt[i-1]]+val[i-1]
                if cand>dp[i][w]: dp[i][w]=cand
    return dp[n][W]
```

**LIS (n log n)**

```python
def lis(a):
    tails=[]
    for x in a:
        # binary search lower_bound in tails
        l,r=0,len(tails)-1; pos=len(tails)
        while l<=r:
            m=(l+r)//2
            if tails[m]>=x:
                pos=m; r=m-1
            else:
                l=m+1
        if pos==len(tails): tails.append(x)
        else: tails[pos]=x
    return len(tails)
```

---

# Module 12 — Greedy & Intervals

* Interval scheduling (max non‑overlapping)
* Merge intervals (manual sort via quicksort)
* Activity selection, gas station, jump game

```python
def can_jump(a):
    reach=0
    for i,x in enumerate(a):
        if i>reach: return False
        if i+x>reach: reach=i+x
    return True
```

---

# Module 13 — Bit Manipulation

* Basic ops, set/clear/test bit
* Count set bits (Brian Kernighan)
* Single number (xor), subset generation from bitmasks

```python
def count_bits(x):
    c=0
    while x:
        x &= x-1
        c+=1
    return c
```

---

## String Algorithms: Rabin–Karp & KMP

```python
def kmp_build(pat):
    pi=[0]*len(pat); j=0
    for i in range(1,len(pat)):
        while j>0 and pat[i]!=pat[j]:
            j=pi[j-1]
        if pat[i]==pat[j]:
            j+=1
            pi[i]=j
    return pi

def kmp_search(txt,pat):
    if not pat: return 0
    pi=kmp_build(pat); j=0
    for i,ch in enumerate(txt):
        while j>0 and ch!=pat[j]:
            j=pi[j-1]
        if ch==pat[j]:
            j+=1
            if j==len(pat):
                return i-j+1
    return -1
```

---

## High‑Yield Problem Sets (with references above)

**Core 30:**

1. Two Sum (custom HT)
2. Valid Anagram w/out sort
3. Longest Substring w/o Repeats (ASCII map)
4. Min Window Substring
5. Max Subarray (Kadane)
6. Product of Array Except Self (prefix/suffix)
7. Merge Intervals (manual quicksort)
8. Insert Interval
9. Non‑overlapping Intervals (greedy)
10. Rotated Array Search (binary search variants)
11. Find First/Last Position (bounds)
12. Kth Largest (manual heap)
13. Top K Frequent Elements (HT + heap)
14. Sort Colors (Dutch flag)
15. Linked List Cycle (Floyd)
16. Reverse Linked List
17. Merge Two Sorted Lists
18. Reorder List (split + reverse + merge)
19. Validate BST
20. LCA in BST
21. Binary Tree Level Order (BFS)
22. Diameter of Binary Tree
23. Implement Trie (optional, add if time)
24. Graph Valid Tree (BFS/DFS/DSU optional)
25. Course Schedule (Topo sort)
26. Number of Islands (DFS/BFS)
27. Clone Graph (BFS with custom HT)
28. Coin Change (DP)
29. Edit Distance (DP)
30. Longest Increasing Subsequence (n log n)

Each of these maps to code in modules above. Add timing and explanation.

---

## Whiteboard/Interview Playbook

* Restate problem + constraints + examples.
* Choose pattern: window / two pointers / hash / stack / tree / graph / DP.
* State complexity target and why.
* Walk through edge cases.
* Write clean code with docstrings and small helpers.
* Test with 2–3 custom cases.
* Reflect on trade‑offs.

**Checklist (memorize):** inputs, outputs, constraints, examples, brute, optimizations, invariants, complexity, tests.

---

## Mock Interviews (Scripts)

**Mock 1 (45 min):**

* Rotated Array Search (10)
* Merge Intervals (15)
* Top K Frequent (20)

**Mock 2 (45 min):**

* Number of Islands (15)
* Kth Smallest in BST (15)
* Coin Change (15)

Rubric: correctness (40), complexity (20), clarity (20), testing (10), time mgmt (10).

---

## Final Gauntlet (Day 29)

15 mixed problems in 120 minutes. Target ≥ 80% pass.

---

## After the Course

* Weekly 10‑problem maintenance set.
* Re‑implement 1 DS/Algo from memory every weekend.
* Keep a log: misses, patterns, fixes.

---

### Notes on “No Libraries” Constraint

* When re‑implementing, do not use: `heapq`, `sorted()`, `list.sort()`, `collections`, `itertools`, `bisect`.
* For character frequency, prefer fixed arrays (`[0]*128`) over dicts.
* For hash‑tables/tries, use your custom implementations above.

---

## Appendix — Practice Template

```python
"""Problem: <name>
Approach: <pattern + invariants>
Complexity: <time, space>
Tests: <key edge cases>
"""

def solve(...):
    pass

# Tests.case("example", lambda: assert_eq(solve(...), ...))
```
