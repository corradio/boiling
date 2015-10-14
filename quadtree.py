'''
Box is given as (xmin, ymin, xmax, ymax) <=> (lower_left, upper_right)
'''

def is_in_box(x, y, box):
    xmin, ymin, xmax, ymax = box
    return x <= xmax and x >= xmin and y <= ymax and y >= ymin

class QuadLeaf:
    def __init__(self, item, x, y):
        self.item = item
        self.x = x
        self.y = y

class QuadNode:
    def __init__(self, box, MAX_ITEMS_PER_NODE = 10):
        self.children = None
        self.leaves = []
        self.box = box
        self.MAX_ITEMS_PER_NODE = MAX_ITEMS_PER_NODE

    def split(self):
        # Create the four quadrants in order to insert the next one
        xmin, ymin, xmax, ymax = self.box
        w = 0.5 * (xmax - xmin)
        h = 0.5 * (ymax - ymin)
        self.children = [
            QuadNode(
                box=(xmin, ymin, xmin + 0.5 * w, ymin + 0.5 * h),
                MAX_ITEMS_PER_NODE=self.MAX_ITEMS_PER_NODE), # bottomleft
            QuadNode(
                box=(xmin + 0.5 * w, ymin, xmax, ymin + 0.5 * h),
                MAX_ITEMS_PER_NODE=self.MAX_ITEMS_PER_NODE), # bottomright
            QuadNode(
                box=(xmin, ymin + 0.5 * h, xmin + 0.5 * w, ymax),
                MAX_ITEMS_PER_NODE=self.MAX_ITEMS_PER_NODE), # topleft
            QuadNode(
                box=(xmin + 0.5 * w, ymin + 0.5 * w, xmax, ymax),
                MAX_ITEMS_PER_NODE=self.MAX_ITEMS_PER_NODE)  # topright
        ]
        # Move and insert
        leaves = self.leaves
        self.leaves = None
        for leaf in leaves: 
            self.insert(leaf.item, leaf.x, leaf.y)

    def insert(self, item, x, y):
        '''
        Returns the node where the insert happened
        '''
        if self.children is not None:
            # Insert current item into proper child
            xmin, ymin, xmax, ymax = self.box
            w = 0.5 * (xmax - xmin)
            h = 0.5 * (ymax - ymin)
            if x < xmin + 0.5 * w and y < ymin + 0.5 * h: # bottomleft
                return self.children[0].insert(item, x, y)
            elif x > xmin + 0.5 * w and y < ymin + 0.5 * h: # bottomright
                return self.children[1].insert(item, x, y)
            elif x < xmin + 0.5 * w and y > ymin + 0.5 * h: # topleft
                return self.children[2].insert(item, x, y)
            elif x >= xmin + 0.5 * w and y >= ymin + 0.5 * h: # topright
                return self.children[3].insert(item, x, y)
            else:
                raise Exception('Unhandled case!')

        elif len(self.leaves) >= self.MAX_ITEMS_PER_NODE:
            # Split and insert into child
            self.split()
            return self.insert(item, x, y)

        else:
            # Keep it at this node
            self.leaves.append(QuadLeaf(item, x, y))
            return self

    def get_within(self, box):
        if self.children is not None:
            return reduce(lambda x, y: x + y,
                map(lambda q: q.get_within(box), self.children))
        else: 
            return map(lambda l: l.item,
                filter(lambda l: is_in_box(l.x, l.y, box), self.leaves))




# class IndexedQuadTree:
#     def __init__(self):
#         self.root = QuadNode()
#         self.items = {} # HashMap of items by key
#         self.nodes = {} # HashMap of nodes by key

#     def insert(self, item, key, x, y):
#         self.items[key] = item
#         self.nodes[key] = self.root.insert(item, x, y)

#     def delete_by_key(self, key):
#         # This does not prune the tree
#         self.nodes[key].delete(self.items[key])
#         del self.items[key]
#         del self.nodes[key]

#     def get_by_key(self, key):
#         return self.items[key]
