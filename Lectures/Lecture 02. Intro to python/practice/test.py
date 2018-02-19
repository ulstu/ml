'''
cascascas
'''
f = lambda x: x ** x
l = [f(i) for i in range(10)]
print (l)
lst = [x for x in l if x % 2 == 0]
print(lst)

