a = 1

def change():
	global a
	a = 10

if __name__ == "__main__":
	print a
	change()
	print a