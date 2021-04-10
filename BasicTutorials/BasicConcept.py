import taichi as ti

ti.init(arch=ti.gpu)

# 分配了一个叫做 pixels 的二维张量，大小是 (640, 320) ，数据类型是 ti.f32
n = 320
pixels = ti.var(dt=ti.f32, shape=(n * 2, n))

# 使用 @ti.func 来修饰 Taichi 函数。这些函数只能在 Taichi 作用域内调用。不要在 Python 作用域内调用它们
# 目前不支持具有多个 return 语句的函数。请用 局部变量 暂存结果，以便最终只有一个 return 语句
# function支持vectors 或 matrices作为参数，kernel不支持
@ti.func
def complex_sqr(z):
    return ti.Vector([z[0] ** 2 - z[1] ** 2, z[1] * z[0] * 2])


# 计算发生在Kernel中，参数必须显式指明类型
# 被修饰的函数体内的代码会通过Taichi编译，其余的都是python本身代码
# Warning: Kernel不支持嵌套，Function可以嵌套；Taichi不支持递归函数!
@ti.kernel
def paint(t: ti.f32):
    for i, j in pixels:  # 对于所有像素，并行执行
        c = ti.Vector([-0.8, ti.sin(t) * 0.2])
        z = ti.Vector([float(i) / n - 1, float(j) / n - 0.5]) * 2
        iterations = 0
        while z.norm() < 20 and iterations < 50:
            z = complex_sqr(z) + c
            iterations += 1
        pixels[i, j] = 1 - iterations * 0.02


gui = ti.GUI("Fractal", (n * 2, n))

for i in range(1000000):
    paint(i * 0.03)
    gui.set_image(pixels)
    gui.show()

# 区间for循环
# Warning：最外层"作用域"的for是并行的，里面的for是顺序的
# @ti.kernel
# def fill():
#   for i in range(10): # 并行执行
#     x[i] += i
#
#     s = 0
#     for j in range(5): # 在每个并行的线程中顺序执行
#       s += j
#
#     y[i] = s
#
# @ti.kernel
# def fill_3d():
#   # 在区间 3 <= i < 8, 1 <= j < 6, 0 <= k < 9 上展开并行
#   for i, j, k in ti.ndrange((3, 8), (1, 6), 9):
#     x[i, j, k] = i + j + k

# Warning：不是最外层循环就是并行的！
# def foo():
#     for i in range(10): # 并行 :-)
#         …
#
# @ti.kernel
# def bar(k: ti.i32):
#     if k > 42:
#         for i in range(10): # 串行 :-(
#             …


# 结构for循环
# 结构 for 循环只能使用在内核的最外层作用域。
# 是最外层 作用域 的循环并行执行，而不是最外层的循环。
# @ti.kernel
# def foo():
#     for i in x:
#         …
#
# @ti.kernel
# def bar(k: ti.i32):
#     # 最外层作用域是 `if` 语句
#     if k > 42:
#         for i in x: # 语法错误。结构 for 循环 只能用于最外层作用域
#             …


# Warning：并行循环不支持 break 语句：
# @ti.kernel
# def foo():
#   for i in x:
#       ...
#       break # 错误：并行执行的循环不能有 break
#
# @ti.kernel
# def foo():
#   for i in x:
#       for j in y:
#           ...
#           break # 可以


# kernel可以有一个 标量 返回值。如果内核有一个返回值，那它必须有类型提示。这个返回值会自动转换到所提示的类型
@ti.kernel
def add_xy(x: ti.f32, y: ti.f32) -> ti.i32:
    return x + y  # 等价于： ti.cast(x + y, ti.i32)


res = add_xy(2.3, 1.1)
print(res)  # 3，因为返回值类型是 ti.i32

# kernel还支持模板参数和外部数组参数
# 不要在可微编程中使用返回值，因为这种返回值并不会被自动微分追踪。取而代之，可以把结果存入全局变量（例如 loss[None]）


# 原子操作
# Warning: 并行修改全局变量时，请确保使用原子操作
# 例如，合计 x 中的所有元素:
@ti.kernel
def sum():
    for i in x:
        # 方式 1: 正确（原子操作）
        total[None] += x[i]

        # 方式 2: 正确
        ti.atomic_add(total[None], x[i])

        # 方式 3: 非原子操作因而会得到错误结果
        total[None] = total[None] + x[i]
# 显式的原子操作（例如 ti.atomic_add ）也可以原子地进行读取-修改-写入
# 这些操作还会返回 第一个参数的 旧值 ：
x[i] = 3
y[i] = 4
z[i] = ti.atomic_add(x[i], y[i])
# 现在 x[i] = 7, y[i] = 4, z[i] = 3

