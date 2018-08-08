from multiprocessing import pool


def test(x):
	return x**2

if __name__=='__main__':
	po = pool.Pool()
	res = po.map_async(test, [i for i in range(10)])
	print (res.get())



def msort(x):
	if len(x)<=1:
		return x
	mid = len(x)//2

	left  = msort(x[:mid])
	right = msort(x[mid:])
	result = []
	while left or right :
		if left and right :
			if left[0] < right[0]:
				result.append(left.pop(0))
			else:
				result.append(right.pop(0))
		if left :
			result.append(left.pop(0))
		if right :
			result.append(right.pop(0))
	return result 