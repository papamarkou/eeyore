def chunk_evenly(iterable, n):
    iterable_len = len(iterable)
    r, a = iterable_len % n, 0
    
    for i, s in enumerate(range(0, iterable_len if (r == 0) else (iterable_len-n), n)):
        yield iterable[(s+a):((s+a+n+1) if (i < r) else (s+a+n))]

        if i < r:
            a = a + 1
