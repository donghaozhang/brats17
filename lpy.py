# Be familiar with zip usage in python
x = [1, 2, 3, 4]
y = [7, 8, 3, 2]
z = ['a', 'b', 'c', 'd']

for a, b in zip(x, y):
    print(a, b)

for a, b, c in zip(x, y, z):
    print(a, b, c)