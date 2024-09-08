import pennylane as qml
import numpy as np
from math import gcd
import random
dev = qml.device("default.qubit", wires=8)
def modular_exponentiation(a, x, N):
    return pow(a, x, N)
def find_order(a, N):
    r = 1
    while (modular_exponentiation(a, r, N) != 1):
        r += 1
    return r
def shors_algorithm(N):
    a = random.randint(2, N-1)
    g = gcd(a, N)
    if g > 1:
        return g, N // g
    r = find_order(a, N)
    if r % 2 != 0 or modular_exponentiation(a, r // 2, N) == N - 1:
        return None
    factor1 = gcd(modular_exponentiation(a, r // 2, N) - 1, N)
    factor2 = gcd(modular_exponentiation(a, r // 2, N) + 1, N)
    return factor1, factor2
def is_prime(N):
    if N <= 1:
        return False
    if N <= 3:
        return True
    if N % 2 == 0:
        return False
    factors = shors_algorithm(N)
    if factors:
        return False
    return True
def check_primes_from(start, limit):
    for number in range(start, limit + 1):
        if is_prime(number):
            print(number)
start_number = int(input('start: '))
input_limit = int(input('end: '))
check_primes_from(start_number, input_limit)