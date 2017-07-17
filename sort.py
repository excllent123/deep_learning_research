'''
Sort 
- bubble_sort 
- selection_sort 
- insertion_sort 
- shell_sort 

- merge_sort 
- quick_sort
'''


def msort(x):
    result = []
    if len(x) < 2:
        return x
    mid = int(len(x)/2)
    y = msort(x[:mid])
    z = msort(x[mid:])
    while (len(y) > 0) or (len(z) > 0):
        if len(y) > 0 and len(z) > 0:
            if y[0] > z[0]:
                result.append(z[0])
                z.pop(0)
            else:
                result.append(y[0])
                y.pop(0)
        elif len(z) > 0:
            for i in z:
                result.append(i)
                z.pop(0)
        else:
            for i in y:
                result.append(i)
                y.pop(0)
        return result


def msort(x):
    if len(x)<2:
        return x
    mid = int(len(x)/2)
    y = msort(x[:mid])
    z = msort(x[mid:])
    res = []
    while y or z :
        if y and z :
            if y[0] < z[0]:
                res.append(y.pop(0))
            else:
                res.append(z.pop(0))
        elif y:
            res.append(y.pop(0))
        else:
            res.append(z.pop(0))
    return res



def mergeSort(alist):
    print("Splitting ",alist)
    if len(alist)>1:
        mid = len(alist)//2
        lefthalf = alist[:mid]
        righthalf = alist[mid:]

        mergeSort(lefthalf)
        mergeSort(righthalf)

        i=0
        j=0
        k=0
        while i < len(lefthalf) and j < len(righthalf):
            if lefthalf[i] < righthalf[j]:
                alist[k]=lefthalf[i]
                i=i+1
            else:
                alist[k]=righthalf[j]
                j=j+1
            k=k+1

        while i < len(lefthalf):
            alist[k]=lefthalf[i]
            i=i+1
            k=k+1

        while j < len(righthalf):
            alist[k]=righthalf[j]
            j=j+1
            k=k+1
    print("Merging ",alist)


def bubble_sort(x):
    '''
    # Description : 

    1. 

    '''
    for i in range(len(x))[::-1]:
        for j in range(i):
            if x[j] > x[j+1]:
                x[j], x[j+1] = x[j+1], x[j]
    return x

def selectionSort(alist):
   for fillslot in range(len(alist)-1,0,-1):
       positionOfMax=0
       for location in range(1,fillslot+1):
           if alist[location]>alist[positionOfMax]:
               positionOfMax = location

       temp = alist[fillslot]
       alist[fillslot] = alist[positionOfMax]
       alist[positionOfMax] = temp


def select_sort(x):
    '''
    Base on Bubble Sort 
    find max position of graduating grow sublist and then changes
    '''
    for i in range(len(x))[::-1]:
        location = 0
        tmp = x[i]
        for j in range(i+1):
            if x[j]>x[location]:
                location = j 
        x[i]= x[location]
        x[location]=tmp
    return x


def insertionSort(alist):
   for index in range(1,len(alist)):

     currentvalue = alist[index]
     position = index

     while position>0 and alist[position-1]>currentvalue:
         alist[position]=alist[position-1]
         position = position-1

     alist[position]=currentvalue


def insertion_sort(x):
    '''
    description :
      - 1. a growing sub-list from [0,1] ot [0, len(list)]
      - 2. for each-candates where is the last element of new sublist
      - 3. reverse sort to compare 
    '''
    for i in range(1, len(x)):
        currentvalue = x[i]
        end_position = i 

        while end_position > 0 and x[end_position-1] > currentvalue:
            x[end_position] = x[end_position-1]
            end_position-=1
        x[end_position] = currentvalue
    return x

# for the most of time, it is always better to manipulate the index of the list



def shell_sort(x):
    '''
    diminishing increment sort 
    sometimes called the gap, to create a sublist by choosing all items that are i items apart.
    '''
    sublist_count = len(x)/2

    pass


def shellSort(alist):
    sublistcount = len(alist)//2 # step 1 : init gap_i 
    while sublistcount > 0:

      for startposition in range(sublistcount):
        gapInsertionSort(alist,startposition,sublistcount)

      print("After increments of size",sublistcount,
                                   "The list is",alist)

      sublistcount = sublistcount // 2

def gapInsertionSort(alist,start,gap):
    for i in range(start+gap,len(alist),gap):

        currentvalue = alist[i]
        position = i

        while position>=gap and alist[position-gap]>currentvalue:
            alist[position]=alist[position-gap]
            position = position-gap

        alist[position]=currentvalue


because it is too hard to express it even in natural-language
so we use mapping & machine learning or ... maybe it is the only right way 

