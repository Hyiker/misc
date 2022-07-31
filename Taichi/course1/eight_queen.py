import taichi as ti
from time import time


N = 12

ti.init(arch=ti.cuda, kernel_profiler=True)

ans_ti = ti.field(ti.i32, shape=())
col_index = ti.field(ti.i32, shape=(N))
stack_ti = ti.Vector.field(2, ti.i32, shape=(N, N+2))
stack_pointer_ti = ti.field(ti.i32, shape=(N))
rows_ti = ti.field(ti.i32, shape=(N, N))
grid_ti = ti.field(ti.i32, shape=(N, N, N))


@ti.func
def stack_clear(i):
    stack_pointer_ti[i] = -1


@ti.func
def stack_push(i, item):
    stack_pointer_ti[i] += 1
    stack_ti[i, stack_pointer_ti[i]] = item


@ti.func
def stack_top(i):
    return stack_ti[i, stack_pointer_ti[i]]


@ti.func
def stack_pop(i):
    top = stack_ti[i, stack_pointer_ti[i]]
    stack_pointer_ti[i] -= 1
    return top


@ti.func
def stack_empty(i):
    return stack_pointer_ti[i] <= -1


@ti.func
def grid_clear(s):
    for i, j in ti.ndrange(N, N):
        grid_ti[s, i, j] = 0


@ti.func
def rows_clear(s):
    for i in range(N):
        rows_ti[s, i] = 0


@ti.kernel
def eight_queen_ti():
    for _i in range(N):
        grid_clear(_i)
        # collection: set[tuple[int, int]] = set()
        rows_clear(_i)
        stack_clear(_i)
        stack_push(_i, ti.Vector([0, _i]))
        grid_ti[_i, 0, _i] = 1
        while not stack_empty(_i):
            li = stack_top(_i)[0]
            i = li + 1
            if i >= N:
                ans_ti[None] += 1
                rows_ti[_i, N - 1] += 1
                tp = stack_pop(_i)
                grid_ti[_i, tp[0], tp[1]] = 0
                continue
            j = rows_ti[_i, i]
            rows_ti[_i, i] += 1
            if j >= N:
                rows_ti[_i, i] = 0
                tp = stack_pop(_i)
                grid_ti[_i, tp[0], tp[1]] = 0
                continue
            flag = False
            a, b = i - 1, j - 1
            while a >= 0 and b >= 0:
                if grid_ti[_i, a, b] == 1:
                    flag = True
                    break
                a -= 1
                b -= 1
            if not flag:
                a, b = i - 1, j
                while a >= 0:
                    if grid_ti[_i, a, b] == 1:
                        flag = True
                        break
                    a -= 1
                a, b = i - 1, j + 1
            if not flag:
                while a >= 0 and b <= N - 1:
                    if grid_ti[_i, a, b] == 1:
                        flag = True
                        break
                    a -= 1
                    b += 1
            if flag:
                continue
            grid_ti[_i, i, j] = 1
            stack_push(_i, ti.Vector([i, j]))


ans_py = 0


def eight_queen_py():
    global ans_py
    for _i in range(N):
        collection: set[tuple[int, int]] = set()
        rows = [0] * N
        stack = []
        stack.append((0, _i))
        collection.add((0, _i))
        while len(stack) > 0:
            tail = stack[-1]
            li, lj = tail
            i = li + 1
            if i >= N:
                ans_py += 1
                rows[N - 1] += 1
                collection.remove(stack.pop())
                continue
            j = rows[i]
            rows[i] += 1
            if j >= N:
                rows[i] = 0
                collection.remove(stack.pop())
                continue
            flag = False
            a, b = i - 1, j - 1
            while a >= 0 and b >= 0:
                if (a, b) in collection:
                    flag = True
                    break
                a -= 1
                b -= 1
            if not flag:
                a, b = i - 1, j
                while a >= 0:
                    if (a, b) in collection:
                        flag = True
                        break
                    a -= 1
                a, b = i - 1, j + 1
            if not flag:
                while a >= 0 and b <= N - 1:
                    if (a, b) in collection:
                        flag = True
                        break
                    a -= 1
                    b += 1
            if flag:
                continue
            collection.add((i, j))
            stack.append((i, j))


def measure(func):
    func()


'''
[Taichi] version 1.0.4, llvm 10.0.0, commit 2827db2c, win, python 3.10.5
[Taichi] Starting on arch=cuda
=========================================================================
Kernel Profiler(count, default) @ CUDA on NVIDIA GeForce RTX 3070 Ti
=========================================================================
[      %     total   count |      min       avg       max   ] Kernel name
-------------------------------------------------------------------------
[ 95.65%   2.467 s      1x | 2467.491  2467.491  2467.491 ms] eight_queen_ti_c56_0_kernel_0_range_for
[  4.34%   0.112 s      1x |  111.924   111.924   111.924 ms] runtime_initialize
[  0.00%   0.000 s      6x |    0.009     0.011     0.016 ms] jit_evaluator_0_kernel_0_serial
[  0.00%   0.000 s      3x |    0.012     0.018     0.030 ms] jit_evaluator_1_kernel_0_serial
[  0.00%   0.000 s      3x |    0.009     0.017     0.032 ms] jit_evaluator_2_kernel_0_serial
[  0.00%   0.000 s      3x |    0.008     0.017     0.031 ms] jit_evaluator_5_kernel_0_serial
[  0.00%   0.000 s      2x |    0.009     0.012     0.015 ms] jit_evaluator_4_kernel_0_serial
[  0.00%   0.000 s      1x |    0.018     0.018     0.018 ms] runtime_memory_allocate_aligned
[  0.00%   0.000 s      1x |    0.014     0.014     0.014 ms] jit_evaluator_3_kernel_0_serial
[  0.00%   0.000 s      1x |    0.011     0.011     0.011 ms] runtime_initialize_snodes
-------------------------------------------------------------------------
[100.00%] Total execution time:   2.580 s   number of results: 10
=========================================================================
ans_ti[None] = 14200
eight_queen_py costing 7.19203 seconds
ans_py = 14200
'''

if __name__ == '__main__':
    eight_queen_ti()
    ti.profiler.print_kernel_profiler_info()
    print(f'{ans_ti[None] = }')

    start = time()
    eight_queen_py()
    end = time()
    print(f'{eight_queen_py.__name__} costing {float(end - start):.5f} seconds')
    print(f'{ans_py = }')
