from multiprocessing import Process, Queue

a = [1, 2, 3]

def f(q):
    print(a)
    q.put([42, None, 'hello'])

if __name__ == '__main__':
    q = Queue()
    p = Process(target=f, args=(q,))
    p.start()
    a = [1, 2, 2]
    print(q.get())    # prints "[42, None, 'hello']"
    p.join()