class Tag(set):
    def add(self, *tags):
        tags = set(tags)
        def decorator(func):
            if hasattr(func, "_tags"):
                func._tags.update(tags)
            else:
                func._tags = tags
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs) if self & func._tags else None
            wrapper._tags = func._tags
            return wrapper
        return decorator

@Tag.add('a')
def test():
    print (1)

@Tag.add('b')
def test2():
    print (2)