# Using recursion to implement power and factorial functions


def power(num, pwr):
    # breaking condition: if we reach zero, return 1
    if pwr == 0:
        return 1
    else:
        return num * power(num, pwr-1)


def factorial(num):
    if (num == 0):
        return 1
    else:
        return num * factorial(num-1)


print("{} hoch {} = {}".format(2, 5, power(2, 5)))
print("{} hoch {} = {}".format(5, 3, power(5, 3)))
print("{}! ergibt {}".format(4, factorial(4)))
print("{}! ergibt {}".format(0, factorial(0)))
