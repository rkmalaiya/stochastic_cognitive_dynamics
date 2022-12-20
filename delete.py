def test_yield():
    for i in range(10):
        yield (i, 1), (2,3)

if __name__ == "__main__":
    arr = [i for (i, _), _ in test_yield()]
    print(arr)