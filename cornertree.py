import math, time, itertools
from itertools import permutations, combinations

print("| Welcome to cornertree.py, counting small permutation patterns !!!")
print("| Developed for Python 2.7.10+, Some routines require NumPy 1.10+")
print("| Chaim Even-Zohar <chaim@ucdavis.edu>, Calvin Leng <cleng@usc.edu>")

#===============================================================================
#   Corner Tree
#===============================================================================

class CornerTree(list):

    def __init__(self, *args):
        "args: child1, child2, ... (CornerTrees)"
        list.__init__(self, args)
        self.corner = '*'

    def __eq__(self, other):
        return self.corner == other.corner and list.__eq__(self, other)
    
    def __repr__(self):
        return self.corner + list.__repr__(self).replace(' ','').replace('[]','')

    def canonicalize(self):
        for child in self:
            child.canonicalize()
        self.sort(key = repr)

    def is_canonical(self):
        for child, next in zip(self, self[1:]):
            if repr(child) > repr(next):
                return False
        for child in self:
            if not child.is_canonical():
                return False
        return True  

class NE(CornerTree):
    def __init__(self, *args):
        CornerTree.__init__(self, *args)
        self.corner = 'NE'

class NW(CornerTree):
    def __init__(self, *args):
        CornerTree.__init__(self, *args)
        self.corner = 'NW'

class SE(CornerTree):
    def __init__(self, *args):
        CornerTree.__init__(self, *args)
        self.corner = 'SE'

class SW(CornerTree):
    def __init__(self, *args):
        CornerTree.__init__(self, *args)
        self.corner = 'SW'

CT = CornerTree

#===============================================================================
#   Sum Tree
#===============================================================================

class SumTree:
    "An array with quick prefix/suffix sum, using a binary tree"

    def __init__(self, n):
        self.length = n
        logn = int(math.ceil(math.log(max(1,n))/math.log(2)))
        self.arrays = [[0]*(2**(logn - x)) for x in range(logn + 1)]
        
    def add(self, i, val):
        for x in range(len(self.arrays)):
            self.arrays[x][int(i / 2**x)] += val  

    def sum_suffix(self, i):
        return sum([self.arrays[x][i/2**x + 1]
                    for x in range(len(self.arrays)-1) if i/2**x % 2 == 0]) 

    def sum_prefix(self, i):
        return sum([self.arrays[x][i/2**x - 1]
                    for x in range(len(self.arrays)-1) if i/2**x % 2 == 1]) 

#===============================================================================
#   Count Occurrences of Corner Trees
#===============================================================================

def count_corner_tree(ct, perm):
    "How many times ct occurs in perm, in near linear time"
    n = len(perm)
    children_count = [1] * n
    for child in ct:
        for t, x in enumerate(count_corner_tree(child, perm)):
            children_count[t] *= x

    if ct.corner == '*':
        return sum(children_count)

    counts = [0] * n
    T = SumTree(n)
    for t in (range(n) if ct.corner[1] == 'W' else
              range(n - 1, -1, -1)):
        counts[t] += (T.sum_suffix(perm[t]) if ct.corner[0] == 'N' else
                      T.sum_prefix(perm[t]))
        T.add(perm[t], children_count[t])

    return counts

#===============================================================================
#   Corner Tree Iterators
#===============================================================================

def adj_pairs(a):
    for x,y in zip(a,a[1:]):
        yield x,y

def partitions(n):
    if n > 0:
        for x in itertools.product(*[[False,True]]*(n-1)):
            bars = [0] + [i for i in range(1,n) if x[i-1]] + [n]
            yield [y-z for z,y in adj_pairs(bars)]
    else:
        yield []
        
def unlabeled_corner_trees(n):
    for p in partitions(n-1):
        for cts in itertools.product(*list(map(unlabeled_corner_trees, p))):
            yield CornerTree(*cts)

def corner_trees(n, root = [NE, NW, SE, SW]):
    for p in partitions(n-1):
        for cts in itertools.product(*list(map(corner_trees, p))):
            for corner in root:
                yield corner(*cts)

def canonical_corner_trees(n, root = [CT], children = [NE, NW, SE, SW], ):
    for p in partitions(n-1):
        for cts in itertools.product(*list(
            [canonical_corner_trees(q, children, children) for q in p])):
            if all([x <= y for x,y in adj_pairs(list(map(repr, cts)))]):
                for corner in root:
                    yield corner(*cts)

#===============================================================================
#   Pattern Expansions
#===============================================================================

def shape(c):
    sc = sorted(c)
    return [sc.index(x) for x in c]

def count_brute_force(pattern, permutation):
    count = 0
    for sigma in list(combinations(permutation, len(pattern))):
        if list(pattern) == shape(sigma):
            count += 1
    return count

def expand_in_patterns(f, k):
    "represent a perm-statistic f by 0..k-pattern counts"
    combos = []
    for i in range(k+1):
        combo = []
        for sigma in permutations(range(i)):
            coef = f(sigma)
            for j in range(0, i):
                for t, tau in enumerate(permutations(range(j))):
                    coef -= count_brute_force(tau, sigma) * combos[j][t]
            combo.append(coef)
        combos.append(combo)
    return combos

def expand_in_patterns_numpy(f, k, cache = {}):
    "represent an *integer* f with 2..k-pattern counts, efficiently"
    import numpy as np
    combos = [[0],[0]]
    for i in range(2,k+1):
        coefs = np.array([f(sigma) for sigma in permutations(range(i))], int)
        for j in range(2, i):
            if (i,j) not in cache:
                cache[i,j] = np.array([[count_brute_force(tau, sigma)
                                        for tau in permutations(range(j))]
                                       for sigma in permutations(range(i))], int)
            coefs -= np.dot(cache[i,j], combos[j])
        combos.append(map(int,coefs))
    return combos

def expand_corner_trees(maxsize = 3,
                        expand_func = expand_in_patterns_numpy,
                        explore_tol = False):
    from numpy.linalg import matrix_rank
    C = []
    for k in range(2, maxsize+1):
        print time.strftime('[%a,%H:%M:%S]') + "\tsize = %d" % k,
        for t in canonical_corner_trees(k, [CT]):
            C.append((t,expand_func(lambda x:count_corner_tree(t,x),maxsize)))
        M,m = [sum(x[1][2:],[]) for x in C], [x[1][k] for x in C]
        print "\ttrees = %d\trank = %d\tnew = %d\t" % (len(C),
                                                       matrix_rank(M),
                                                       matrix_rank(m)), \
              ' '.join([str(matrix_rank(M, tol = 10**-t))
                        for t in range(20)]) if explore_tol else ''
    print time.strftime('[%a,%H:%M:%S]'), '\tdone'
    return C

#===============================================================================
#   Corner Tree Formulas
#===============================================================================

class CornerTreeFormula:

    def __init__(self, *terms):
        "terms: (coef1, tree1), (coef2, tree2), ... "
        self.terms = terms

    def __repr__(self):
        return ' + '.join(['(%s)%s' % (c,t) for c,t in self.terms])

    def __call__(self, perm, roundit = True):
        s = sum([count_corner_tree(t, perm) * c for c,t in self.terms], 0)
        return int(round(s)) if roundit else s

CTF = CornerTreeFormula

CTFLIB = {
    (0,1) : CTF((1.0,CT(NE()))),
    (1,0) : CTF((1.0,CT(SE()))),
    (0,1,2) : CTF((1.0,CT(NE(NE())))),
    (0,2,1) : CTF((0.5,CT(NE(),NE())), (-1.0, CT(NE(NE()))), (-0.5, CT(NE()))),
    (1,0,2) : CTF((0.5,CT(SW(),SW())), (-1.0, CT(SW(SW()))), (-0.5, CT(SW()))),
    (1,2,0) : CTF((0.5,CT(NW(),NW())), (-1.0, CT(NW(NW()))), (-0.5, CT(NW()))),
    (2,0,1) : CTF((0.5,CT(SE(),SE())), (-1.0, CT(SE(SE()))), (-0.5, CT(SE()))),
    (2,1,0) : CTF((1.0,CT(SE(SE())))),
    (0,1,2,3) : CTF((1.0,CT(NE(NE(NE()))))),
    (3,2,1,0) : CTF((1.0,CT(SE(SE(SE()))))),
    (0,1,3,2) : CTF((0.5,CT(NE(NE(),NE()))), (-1.0, CT(NE(NE(NE())))), (-0.5, CT(NE(NE())))),
    (1,0,2,3) : CTF((0.5,CT(SW(SW(),SW()))), (-1.0, CT(SW(SW(SW())))), (-0.5, CT(SW(SW())))),
    (2,3,1,0) : CTF((0.5,CT(NW(NW(),NW()))), (-1.0, CT(NW(NW(NW())))), (-0.5, CT(NW(NW())))),
    (3,2,0,1) : CTF((0.5,CT(SE(SE(),SE()))), (-1.0, CT(SE(SE(SE())))), (-0.5, CT(SE(SE())))),
    (1,0,3,2) : CTF((1.0,CT(SE(),NE(SE()))),
                    (-1.0,CT(SE(),SE(SW()))),
                    (1.0,CT(SE(),SE(NE()))),
                    (-2.0,CT(SW(SE(SE())))),
                    (-2.0,CT(SW(SE(SW())))),                    
                    (-1.0,CT(SW(SE()))),                    
                    (-1.0,CT(NE(NE(),NE()))),
                    (1/3., CT(NE(),NE(),NE())),
                    (-0.5,CT(NE(),NE())),
                    (1/6., CT(NE()))),                    
    (2,3,0,1) : CTF((1.0,CT(NE(),SE(NE()))),
                    (-1.0,CT(NE(),NE(NW()))),
                    (1.0,CT(NE(),NE(SE()))),
                    (-2.0,CT(NW(NE(NE())))),
                    (-2.0,CT(NW(NE(NW())))),                    
                    (-1.0,CT(NW(NE()))),                    
                    (-1.0,CT(SE(SE(),SE()))),
                    (1/3., CT(SE(),SE(),SE())),
                    (-0.5,CT(SE(),SE())),
                    (1/6., CT(SE()))),                    
    }

def test_formulas(n = 10, lib = CTFLIB):
    perm = range(n)
    import random
    random.shuffle(perm)
    print "testing", perm
    for pattern, formula in lib.items():
        a = count_brute_force(pattern, perm)
        b = formula(perm)
        s = str(formula)
        print '%12s %5s %8s %8s   %s' % (pattern, a==b, a, b,
                                           s[:55]+('...' if len(s)>55 else ''))
    return perm

#===============================================================================
#   Bergsma Dassios Yanagimoto
#===============================================================================

CTF_S = CTF((2,CT(SE(),NE(),NE())),
            (2,CT(NE(SE(NE())))),
            (-2,CT(NE(),SE(NE()))),
            (-1,CT(NE(),SE())))

SYM_S = CTF(
    (1/2., CT(SE(NE(SE())))),   (1/2., CT(NE(SE(NE())))),
    (1/2., CT(SE(SW(SE())))),   (1/2., CT(SW(SE(SW())))),
    (1/4., CT(SE(),SE(),SW())), (1/4., CT(SE(),SW(),SW())),
    (1/4., CT(NE(),NE(),NW())), (1/4., CT(NE(),NW(),NW())),
    (1/4., CT(SE(),SE(),NE())), (1/4., CT(SE(),NE(),NE())),
    (1/4., CT(NW(),NW(),SW())), (1/4., CT(NW(),SW(),SW())),
    (-1/2., CT(SE(NE(NW())))),  (-1/2., CT(NE(NW(SW())))),
    (-1/2., CT(NW(SW(SE())))),  (-1/2., CT(SW(SE(NE())))),
    (-1/4., CT(SE(),SW())),     (-1/4., CT(NE(),NW())),
    (-1/4., CT(SE(),NE())),     (-1/4., CT(SW(),NW())))

def revperm(p):
    return p[::-1]

def cmpperm(p):
    return [len(p)-1-x for x in p]

def invperm(p):
    d = {y:x for x,y in enumerate(p)} 
    return [d[y] for y in range(len(p))]

def BergsmaDassiosYanagimoto_1(perm):
    return sum([CTFLIB[pattern](perm) for pattern in
                [(0,1,2,3),(0,1,3,2),(1,0,2,3),(1,0,3,2),(3,2,1,0),(2,3,1,0),(3,2,0,1),(2,3,0,1)]])

def BergsmaDassiosYanagimoto_2(perm):
    n = len(perm)
    return n*(n-1)*(n-2)*(n-3)/24 - sum([CTF_S(f3(f2(f1(perm))))
                                         for f1 in (lambda p:p,revperm)
                                         for f2 in (lambda p:p,cmpperm)
                                         for f3 in (lambda p:p,invperm)]) / 8

def BergsmaDassiosYanagimoto_3(perm):
    n = len(perm)
    return n*(n-1)*(n-2)*(n-3)/24 - SYM_S(perm)

def BergsmaDassiosYanagimoto_4(perm):
    n = len(perm)
    S = 0
    for p in [q[::t] for q in [perm,invperm(perm)] for t in [1,-1]]:
        A,Au,Ad,Aud = SumTree(n),SumTree(n),SumTree(n),SumTree(n)      
        for x in range(n):
            u = A.sum_prefix(p[x])
            d = A.sum_suffix(p[x])
            du = Ad.sum_prefix(p[x])
            ud = Au.sum_suffix(p[x])
            udu = Aud.sum_prefix(p[x])
            A.add(p[x],1)
            Au.add(p[x],u)
            Ad.add(p[x],d)
            Aud.add(p[x],ud)
            S += 2*udu-d*du-u*ud+u*d*(x-1)
    return n*(n-1)*(n-2)*(n-3)/24 - S/4
        
Tstar = BergsmaDassiosYanagimoto_4

#===============================================================================
#   Product Tree
#===============================================================================

def dyadic_range(l, r):
    x = 1
    while l != r:
        if l % (2*x) != 0:
            yield (l,l+x)
            l += x
        if r % (2*x) != 0:
            yield (r-x,r)
            r -= x
        x *= 2

class ProductTree:
    "A 2-dim array with quick box sum"
    
    def __init__(self, n):
        self.length = n
        self.logn = int(math.ceil(math.log(max(1,n))/math.log(2)))
        self.table = {}

    def add(self, x, y, value):
        for i in range(self.logn + 1):
            for j in range(self.logn + 1):
                key = (x-x%2**i, x-x%2**i+2**i,
                       y-y%2**j, y-y%2**j+2**j)
                self.table[key] = self.table.get(key,0) + value

    def sum_box(self, x1, x2, y1, y2):
        count = 0
        for x_range in dyadic_range(x1, x2):
            for y_range in dyadic_range(y1, y2):
                count += self.table.get(x_range + y_range, 0)
        return count

#===============================================================================
#   3 2 1 4
#===============================================================================

def count_3214(perm):
    n = len(perm)
    chunk = int(n**(1/3.))
    permT = invperm(perm)
    count = 0

    # if 3,4 not in same row
    for row in range(0, n, chunk):
        T3 = SumTree(row)
        T32 = SumTree(row)
        N321 = 0
        for y in perm:
            if y < row:
                T3.add(y,1)
                T32.add(y,T3.sum_suffix(y))
                N321 += T32.sum_suffix(y)
            if row <= y < row+chunk:
                count += N321
                
    # if 1,4 not in same col, but 3,4 in same row
    for col in range(0, n, chunk):
        T1 = SumTree(col)
        T21 = SumTree(col)
        N321 = 0
        for y,x in enumerate(permT):
            if y % chunk == 0:
                N321back = N321 
            if x < col:
                T1.add(x,1)
                T21.add(x,T1.sum_suffix(x))
                N321 += T21.sum_suffix(x)
            if col <= x < col+chunk:
                count += N321 - N321back
                
    # if both are
    PT = ProductTree(n)
    for x2,y2 in enumerate(perm):
        PT.add(x2,y2,1)
    for x4,y4 in enumerate(perm):
        col,row = x4-x4%chunk,y4-y4%chunk
        for x1,y1 in enumerate(perm[col:x4], col):
            for y3,x3 in enumerate(permT[row:y4], row):
                if x3 < x1 and y3 > y1:
                    count += PT.sum_box(x3+1,x1,y1+1,y3)
                
    return count

#===============================================================================
#   3 2 4 1
#===============================================================================

def count_3241(perm):
    n = len(perm)
    chunk = int(n**(1/2.))
    ch2 = lambda z:z*(z-1)/2
    count = 0

    # if 3,4 or 2,1 not in the same row
    for row in range(0, n, chunk):
        A1 = SumTree(n)
        A12 = SumTree(n)
        B1 = SumTree(row)
        B21 = SumTree(row)
        C = 0
        for y in perm:
            if y < row:
                B1.add(y,1)
                B21.add(y,B1.sum_suffix(y))
                C += B1.sum_suffix(y)*(y-B1.sum_prefix(y))-B21.sum_suffix(y)
            if y >= row + chunk:
                A1.add(y,1)
                A12.add(y,A1.sum_prefix(y))
                C += ch2(A1.sum_prefix(y))-A12.sum_prefix(y)
                C -= ch2(A1.sum_prefix(y-y%chunk))-A12.sum_prefix(y-y%chunk)
            if row <= y < row+chunk:
                count += C

    # if both are
    PT = ProductTree(n)
    permT = invperm(perm)
    for y,x in enumerate(permT):
        for z in permT[y+1:y-y%chunk+chunk]:
            if z > x:
                count += PT.sum_box(x+1,z,z+1,n)
        for z in permT[y-y%chunk:y]:
            if z > x:
                PT.add(x,z,1)
                
    return count

#===============================================================================
#   Examples
#===============================================================================

def usage():
    print "===== SOME EXAMPLES FOR FUNCTIONS USAGE ====="
    for line in ["CT(NW(NE(NE())),NE(SW(),SE(),SW()))",
                 "count_corner_tree(CT(SW()), [3,4,0,1,2])",
                 "count_brute_force([0,1], [3,4,0,1,2])",
                 "count_corner_tree(CT(SW(SW())), [3,4,0,1,2])",
                 "count_corner_tree(CT(SW(),SW()), [3,4,0,1,2])",
                 "expand_in_patterns(lambda x:count_brute_force([0,1],x), 3)",
                 "expand_in_patterns(lambda x:count_corner_tree(CT(NE()),x), 3)",
                 "expand_in_patterns(lambda x:count_corner_tree(CT(NE(SE())),x), 3)",
                 "expand_in_patterns(lambda x:count_corner_tree(CT(NE(),NE()),x), 3)",
                 "expand_in_patterns(len, 3)",
                 "expand_in_patterns(lambda x:7, 3)",
                 "list(canonical_corner_trees(2, [CT]))",
                 "list(canonical_corner_trees(3, [CT]))",
                 "expand_corner_trees(4)[100]",
                 "CornerTreeFormula((1/2.,CT(NE(),NE())),(-1,CT(NE(NE()))),(-1/2.,CT(NE())))",
                 "CTF((1.0, CT(NE())), (-1.0, CT(SE())))([3,4,0,1,2])",
                 "CTFLIB[1,0,3,2]",
                 "CTFLIB[1,0,3,2]([1,0,3,2,5,4,6])",
                 "'tested: %s' % test_formulas(23, CTFLIB)",
                 "CTF_S",
                 "SYM_S",
                 "expand_in_patterns(SYM_S,4)",
                 "BergsmaDassiosYanagimoto_1([4,0,6,9,2,8,1,3,7,5])",
                 "BergsmaDassiosYanagimoto_2([4,0,6,9,2,8,1,3,7,5])",
                 "BergsmaDassiosYanagimoto_3([4,0,6,9,2,8,1,3,7,5])",
                 "BergsmaDassiosYanagimoto_4([4,0,6,9,2,8,1,3,7,5])",
                 "[Tstar(x) for x in permutations(range(4))]",
                 "'tested: %s' % test_formulas(22, {(2,1,0,3):count_3214, (2,1,3,0):count_3241})",
                 ]:
        print ">>>", line
        print eval(line)
    print "============================================="
