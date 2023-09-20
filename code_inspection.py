import inspect

def test(test):
    print(test)


source_code = inspect.getsource(test)
print(source_code)
