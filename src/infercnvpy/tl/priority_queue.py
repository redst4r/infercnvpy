from collections import namedtuple
"""
Priority Queue implemented via a binary Heap
mstly from https://www.geeksforgeeks.org/priority-queue-using-binary-heap/
"""

Element = namedtuple("HeapElement", "priority datum")

# Function to return the index of the
# parent node of a given node
def parent(i) :
 
    return (i - 1) // 2
   
# Function to return the index of the
# left child of the given node
def leftChild(i) :
 
    return ((2 * i) + 1)
   
# Function to return the index of the
# right child of the given node
def rightChild(i) :
 
    return ((2 * i) + 2)
     

class PriorityHeap():
    def __init__(self, max_size, verbose=False):
        self.H = [Element(0, None) for _ in range(max_size)]  #[(0, None) for _ in range(max_size)]
        self.size= -1  #TODO weird, why -1
        self.max_size = max_size  #TODO redundant wuth len(H)
        self.verbose=verbose
        
    # Function to shift up the 
    # node in order to maintain 
    # the heap property
    def shiftUp(self, i) :
        assert self._exists(i), f"element at node {i} does not exist"
        while (i > 0 and self.H[parent(i)].priority < self.H[i].priority) :

            # Swap parent and current node
            self.swap(parent(i), i)

            # Update i to parent of i
            i = parent(i)

    # Function to shift down the node in
    # order to maintain the heap property
    def shiftDown(self, i) :
        assert self._exists(i), f"element at node {i} does not exist"
        maxIndex = i

        # Left Child
        l = leftChild(i)

        if (l <= self.size and self.H[l].priority > self.H[maxIndex].priority) :

            maxIndex = l

        # Right Child
        r = rightChild(i)

        if (r <= self.size and self.H[r].priority > self.H[maxIndex].priority) :

            maxIndex = r

        # If i not same as maxIndex
        if (i != maxIndex) :

            self.swap(i, maxIndex)
            self.shiftDown(maxIndex)

    # Function to insert a 
    # new element in 
    # the Binary Heap
    def insert(self, priority, datum) :
        if self.verbose:
            pass #print(f"HEAP\tInserting {datum} -- {priority}")
        if self.size + 1 >= self.max_size:  # TODO an empty Heap has size -1!!
            raise ValueError(f"heap is at max capacity {self.max_size}")
        self.size +=1 
        self.H[self.size] = Element(priority, datum)

        # Shift Up to maintain 
        # heap property
        self.shiftUp(self.size)

    # Function to extract 
    # the element with
    # maximum priority
    def extractMax(self) :
        if self.size <0 :  # empty heap has size -1
            raise ValueError("Heap empty")
        result = self.H[0]

        if self.size == 0:  # there's only one element in the tree (the max we're about to extract
            self.H[0] = Element(0, None) 
            self.size = -1
            return result  # careful here, result == H[0], but we changed H[0] to empty above. It works (result point to the actual object, not the pointer in the list)
        else:
            # Replace the value 
            # at the root with 
            # the last leaf
            self.H[0] = self.H[self.size]
            self.H[self.size] = Element(0, None) ## fill the old element
            self.size -=  1

            # Shift down the replaced 
            # element to maintain the 
            # heap property
            self.shiftDown(0)
            return result

    # Function to change the priority
    # of an element
    def changePriority(self, i,p) :
        assert self._exists(i), f"element at node {i} does not exist"
        oldp = self.H[i].priority
        
        # doesnt work due to immutable tuples
        #self.H[i].priority = p
        self.H[i] = Element(p, self.H[i].datum)

        if self.verbose:
            print(f"HEAP\tUpdating {self.H[i].datum} -- {oldp} -> {p}")
        
        if (p > oldp) :
            self.shiftUp(i)
        else:
            self.shiftDown(i)

    # Function to get value of 
    # the current maximum element
    def getMax(self,) :

        return self.H[0]

    # Function to remove the element
    # located at given index
    def Remove(self, i) :
        """
        essetnaiylly, make it (artificially) the most important node, bubble it to the top and pop
        """
        assert self._exists(i), f"element at node {i} does not exist"
        
        # artificially increase priority
        newP = self.getMax().priority + 1
        
        # since tuples are immutable this doesnt work:
        #self.H[i].priority = newP
        self.H[i] = Element(newP, self.H[i].datum)
    
        if self.verbose:
            print(f"HEAP\tRemoving {self.H[i].datum}")
        
        # Shift the node to the root
        # of the heap
        self.shiftUp(i)

        # Extract the node
        self.extractMax()

    def swap(self, i, j) :

        temp = self.H[i]
        self.H[i] = self.H[j]
        self.H[j] = temp
        
    def _exists(self, i):
        if i > self.size:
            return False
        return Element(0, None) != self.H[i]
    
    
    def search_element(self, datum):
        """
        descend the tree level by level,
        """
        current_level_nodes = [0]
        while len(current_level_nodes) > 0:
            next_level_nodes = []
            for current_node in current_level_nodes:
                if self.H[current_node].datum == datum:
                    return current_node
                
                lC = leftChild(current_node)
                rC = rightChild(current_node)
                if self._exists(lC):
                    next_level_nodes.append(lC)
                if self._exists(rC):
                    next_level_nodes.append(rC)
            current_level_nodes = next_level_nodes
        
        raise ValueError(f"Element not found {datum}")

