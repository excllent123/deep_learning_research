'''
# Segment Tree 
  - A binary Tree 
  - A binary Tree could bring extra-information such as max, mean, sum, ... etc 

# Usage :
  -

# Hint : 
  - for recusive : 
    End-Point Condition, Breaking Condition, Boudary Condition   
    Condition Set 
'''



class Node(object):
    def __init__(self, start, end):
        self.start = start
        self.end   = end 
        self.left  = None
        self.right = None 

class SegmentTree():
    def __init__(self, x):
        assert type(x)==list 
        self.root = self.build(0, len(x))

    def build(self, start, end):
        if start > end:
            return None
        node = Node(start, end)
        if start ==end:
            return node
        mid = (start+mid)/2
        node.left = self.build(start, mid)
        node.right = self.build(mid+1, end)
        return node

class Solution: 
    # @param start, end: Denote an segment / interval
    # @return: The root of Segment Tree
    def build(self, start, end):
        if start > end:
            return None
        root = SegmentTreeNode(start, end)
        if start == end:
            return root
        root.left = self.build(start, (start + end) / 2)
        root.right = self.build((start + end) / 2 + 1, end)
        return root