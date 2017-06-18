

	def isSymmetric(self,root):
		if not root:
			return False	

		l = root.left
		r = root.right 
		return self.compare(l,r)	

	def compare(self, l , r):
		if not l and not r:
			return True
		if l and r:
			if l.val == r.val:
				return self.compare(l.left, r.right) and self.compare(l.right, r.left)
		return False
		
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <ctime>
#include <cassert>

