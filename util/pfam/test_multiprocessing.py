import os
import argparse
import multiprocessing as mp


def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())


def f(q):
    info('function f')
    q.put('hello')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test multiprocessing.')
    parser.add_argument(
        '-m',
        '--mp',
        type=str,
        default='fork',
        choices=['fork', 'spawn', 'forkserver'],
        help="mp method. (default: %(default)s)"
    )
    args, unparsed = parser.parse_known_args()
    info('main line')
    ctx = mp.get_context(args.mp)
    q = ctx.Queue()
    p = ctx.Process(target=f, args=(q, ))
    p.start()
    print(q.get())
    p.join()
