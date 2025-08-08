# **Python Course: Data Structures & Algorithms for Coding Interviews (No Built-in Libraries)**

### **Introduction**

In coding interviews, it is common to be restricted from using built-in libraries or methods like `heapq`, `sorted()`, or other high-level abstractions. This course is designed to ensure that you understand how to implement data structures and algorithms from scratch using Python, and how to approach coding problems with fundamental programming concepts.

---

## **Module 1: Introduction to DSA and Coding Interviews**

### **Lesson 1.1: Introduction to Data Structures and Algorithms**

* **Data Structures**: Arrays, Linked Lists, Trees, Graphs, Stacks, Queues, Heaps, Hash Tables.

* **Algorithms**: Sorting, Searching, Dynamic Programming, Greedy Algorithms, Backtracking, Divide and Conquer.

* **Big O Notation**: Understand time and space complexity to assess the performance of your code.

  * **O(1)**: Constant time.
  * **O(n)**: Linear time.
  * **O(log n)**: Logarithmic time.
  * **O(n^2)**: Quadratic time.

### **Lesson 1.2: What to Expect in Coding Interviews**

* Problem-solving is key. You will be asked to implement data structures or algorithms without relying on built-in libraries.
* Be clear, concise, and demonstrate the ability to write optimal code that works under time constraints.

---

## **Module 2: Arrays & Strings**

### **Lesson 2.1: Introduction to Arrays**

* An array is a collection of elements, indexed by position.

```python
# Implementing an array manually
class MyArray:
    def __init__(self):
        self.arr = []
    
    def append(self, value):
        self.arr += [value]
    
    def remove(self, value):
        self.arr = [x for x in self.arr if x != value]
    
    def get(self, index):
        return self.arr[index]
    
    def length(self):
        return len(self.arr)
```

### **Lesson 2.2: Common Array Problems**

#### Problem 1: Find the Largest/Smallest Element

```python
def find_largest(arr):
    largest = arr[0]
    for i in arr:
        if i > largest:
            largest = i
    return largest

def find_smallest(arr):
    smallest = arr[0]
    for i in arr:
        if i < smallest:
            smallest = i
    return smallest
```

#### Problem 2: Move Zeros to the End

```python
def move_zeros(arr):
    non_zero = [i for i in arr if i != 0]
    zeros = [0] * (len(arr) - len(non_zero))
    return non_zero + zeros
```

#### Problem 3: Reverse an Array

```python
def reverse_array(arr):
    return arr[::-1]  # Avoid built-in reverse function
```

### **Lesson 2.3: Strings**

#### Problem 1: Reverse a String

```python
def reverse_string(s):
    result = ''
    for i in range(len(s) - 1, -1, -1):
        result += s[i]
    return result
```

#### Problem 2: Check for Anagram

```python
def is_anagram(str1, str2):
    if len(str1) != len(str2):
        return False
    count = {}
    for char in str1:
        count[char] = count.get(char, 0) + 1
    for char in str2:
        if count.get(char, 0) == 0:
            return False
        count[char] -= 1
    return True
```

---

## **Module 3: Linked Lists**

### **Lesson 3.1: Introduction to Linked Lists**

A linked list is a collection of nodes, each holding a value and a reference to the next node.

```python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        last = self.head
        while last.next:
            last = last.next
        last.next = new_node
```

### **Lesson 3.2: Linked List Problems**

#### Problem 1: Detect Cycle in a Linked List

```python
def has_cycle(head):
    slow, fast = head, head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False
```

#### Problem 2: Reverse a Linked List

```python
def reverse_linked_list(head):
    prev = None
    current = head
    while current:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node
    return prev
```

---

## **Module 4: Stacks & Queues**

### **Lesson 4.1: Introduction to Stacks**

A stack operates on a Last In, First Out (LIFO) principle.

```python
class Stack:
    def __init__(self):
        self.stack = []

    def push(self, value):
        self.stack.append(value)

    def pop(self):
        if self.stack:
            return self.stack.pop()
        return None

    def peek(self):
        return self.stack[-1] if self.stack else None
```

#### Problem: Balancing Parentheses

```python
def is_balanced(s):
    stack = []
    for char in s:
        if char == '(':
            stack.append(char)
        elif char == ')':
            if not stack:
                return False
            stack.pop()
    return len(stack) == 0
```

### **Lesson 4.2: Introduction to Queues**

A queue operates on a First In, First Out (FIFO) principle.

```python
class Queue:
    def __init__(self):
        self.queue = []

    def enqueue(self, value):
        self.queue.append(value)

    def dequeue(self):
        if self.queue:
            return self.queue.pop(0)
        return None

    def front(self):
        return self.queue[0] if self.queue else None
```

---

## **Module 5: Trees & Graphs**

### **Lesson 5.1: Introduction to Trees**

A tree is a hierarchical data structure consisting of nodes, with a root and child nodes.

#### Binary Search Tree (BST) Example:

```python
class BSTNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

def insert_bst(root, value):
    if not root:
        return BSTNode(value)
    if value < root.value:
        root.left = insert_bst(root.left, value)
    else:
        root.right = insert_bst(root.right, value)
    return root
```

#### Traversal Methods

```python
def inorder(root):
    if root:
        inorder(root.left)
        print(root.value, end=" ")
        inorder(root.right)
```

### **Lesson 5.2: Graph Traversal**

Graphs can be represented using an adjacency list or matrix. DFS and BFS are common traversal methods.

#### Depth First Search (DFS)

```python
def dfs(graph, node, visited=None):
    if visited is None:
        visited = set()
    visited.add(node)
    for neighbor in graph.get(node, []):
        if neighbor not in visited:
            dfs(graph, neighbor, visited)
    return visited
```

---

## **Module 6: Heaps & Hashing**

### **Lesson 6.1: Introduction to Heaps**

A heap is a complete binary tree that satisfies the heap property.

```python
class MinHeap:
    def __init__(self):
        self.heap = []

    def insert(self, value):
        self.heap.append(value)
        self._heapify_up()

    def _heapify_up(self):
        index = len(self.heap) - 1
        while index > 0:
            parent_index = (index - 1) // 2
            if self.heap[index] < self.heap[parent_index]:
                self.heap[index], self.heap[parent_index] = self.heap[parent_index], self.heap[index]
                index = parent_index
            else:
                break

    def extract_min(self):
        if len(self.heap) == 1:
            return self.heap.pop()
        root = self.heap[0]
        self.heap[0] = self.heap.pop()
        self._heapify_down()
        return root

    def _heapify_down(self):
        index = 0
        while 2 * index + 1 < len(self.heap):
            left = 2 * index + 1
            right = 2 * index + 2
            smallest = left
            if right < len(self.heap) and self.heap[right] < self.heap[left]:
                smallest = right
            if self.heap[index] > self.heap[smallest]:
                self.heap[index], self.heap[smallest] = self.heap[smallest], self.heap[index]
                index = smallest
            else:
                break
```

---

## **Module 7: Dynamic Programming**

### **Lesson 7.1: Introduction to Dynamic Programming**

Dynamic Programming (DP) is used for solving optimization problems by breaking them down into smaller subproblems.

#### Fibonacci Sequence (Top-Down Approach)

```python
def fibonacci(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fibonacci(n - 1, memo) + fibonacci(n - 2, memo)
    return memo[n]
```

---

### **Conclusion**

By practicing and implementing data structures and algorithms from scratch, you will be better prepared for coding interviews, where restrictions are often placed on using built-in libraries. Focus on solving problems using fundamental techniques and always strive for an efficient and clean solution.
