#Freda Osei 401 Assignment 1
def flatten(List):
    final = []
    for each_list in List:
        for each_number in each_list:
            final.append(each_number)
    return final
lists = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10,11,12]]


def powerset(List):
    power = [[]]
    for number in List:
        power.extend([x + [number] for x in power])
    return power

def all_perms(List):
    if List:
        first = []
        second =[]
        for x in List:
            if x not in second:
                rest = List[:];
                rest.remove(x)
                for y in all_perms(rest):
                    first.append([x]+y)
            second.append(x)
        return first
    else:
        return [[]]

def spiral(n, end_corner):
	distance = 0
	bottom = 0
	side = n
	coord = [(0,-1),(1,0),(0,1),(-1,0)]
	if end_corner == 1:
		bottom = -1
		side = 0
		coord = coord[1:] + [coord[0]]
	if end_corner == 3:
		bottom = n-1
		side = -1
		coord = coord[2:] + coord[0:2]
	if end_corner == 4:
		bottom = n
		side = n-1
		coord = coord[3:] + coord[0:3]

	result = []
	for i in range(0, n):
		result.append([])
		for j in range(0,n):
			result[i].append(-1)

	number = n
	endMinus1 = n**2
	wayCount = 1
	while endMinus1 > 0:
		bottom += coord[0][0]
		side += coord[0][1]
		endMinus1 -= 1
		result[bottom][side]  = endMinus1
		distance += 1
		if distance == number:
			distance = 0
			wayCount += 1
			coord = coord[1:] + [coord[0]]
			if wayCount == 2:
				wayCount = 0
				number -= 1

	for x in range(0,n):
		print (result[x])

print("flatten")
print(flatten(lists))
print("powerset")
print(powerset((1,2,3)))
print("print permutation")
print(all_perms([1,2,3]))
print("print spiral")
print(spiral(8,2))
