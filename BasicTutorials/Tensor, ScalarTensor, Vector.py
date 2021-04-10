# 由标量组成的张量
# 每个全局变量都是个N维张量。
#   全局 标量 被视为标量的0-D张量。

# 总是使用索引访问张量
#   例如，如果 x 是标量3D张量，则 x[i, j, k] 。
#   即使访问0-D张量 x ，也应使用 x[None] = 0 而不是 x = 0 请 始终 使用索引访问张量中的条目
# 张量元素全部会被初始化为0
# 稀疏张量的元素最初是全部未激活的


# 标量 scalar_tensor
ti.var(dt, shape = None, offset = None)
# dt – （数据类型）张量元素的数据类型
# shape – （可选，标量或元组）张量的形状
# offset – （可选，标量或元组）坐标轴移动，原点可以不是0

# 例如，这将创建一个具有四个 int32 作为元素的 稠密(dense) 张量：
x = ti.var(ti.i32, shape=4)

# 这将创建一个元素为 float32 类型的4x3 稠密 张量：
x = ti.var(ti.f32, shape=(4, 3))

# 如果 shape 是 () （空元组），则创建一个0-D张量（标量）：
x = ti.var(ti.f32, shape=())
# 随后通过传递 None 作为索引来访问它：
x[None] = 2

# 如果形状参数 未提供 或指定为 None，则其后用户必须在手动放置 (place) 它：
x = ti.var(ti.f32)
ti.root.dense(ti.ij, (4, 3)).place(x)
# 等价于: x = ti.var(ti.f32, shape=(4, 3))

# 在任何内核调用或变量访问之前，所有变量都必须被创建和放置完毕（类似古老的C语言，总之在创建都需要在开始前，不可以中途声明）
# Case1:
x = ti.var(ti.f32)
x[None] = 1 # 错误：x没有放置!

# Case2:
x = ti.var(ti.f32, shape=())
@ti.kernel
def func():
    x[None] = 1

func()
y = ti.var(ti.f32, shape=())# 错误：内核调用后不能再创建新的变量!

# Case3:
x = ti.var(ti.f32, shape=())
x[None] = 1
y = ti.var(ti.f32, shape=())# 错误：任一变量访问过后不能再创建新的变量!


# 访问分量
# 通过一个或多个索引来访问 Taichi 张量的元素
a[p, q, ...]
# 参数:	
# a – （张量）标量张量
# p – 第一个张量维度的（标量）索引
# q – 第二个张量维度的（标量）索引
# 返回:	
# （标量） [p, q, ...] 处的元素；如果 a 是由 Vector / Matrix 构成的张量，则返回的值也可以是 Vector / Matrix

# 元数据
    a.shape
    # 参数:	a – （张量）张量
    # 返回:	（元组）张量 a 的形状
    x = ti.var(ti.i32, (6, 5))
    x.shape  # (6, 5)

    y = ti.var(ti.i32, 6)
    y.shape  # (6,)

    z = ti.var(ti.i32, ())
    z.shape  # ()


    a.dtype
    # 参数:	a – （张量）张量
    # 返回:	（数据类型） a 的数据类型
    x = ti.var(ti.i32, (2, 3))
    x.dtype  # ti.i32


    a.parent(n = 1)
    # 参数:	
    # a – （张量）张量
    # n – （可选，标量）父级步数，即父级节点为 n = 1，祖父级节点为 n = 2，等等。
    # 返回:	
    # （结构节点） a 所属结构节点的父类节点
    x = ti.var(ti.i32)
    y = ti.var(ti.i32)
    blk1 = ti.root.dense(ti.ij, (6, 5))
    blk2 = blk1.dense(ti.ij, (3, 2))
    blk1.place(x) # 类似x = ti.var(ti.i32, shape=(6, 5))
    blk2.place(y) # 类似y = ti.Matrix(3, 2, dt=ti.i32, shape=(6,5))

    x.parent()   # blk1
    y.parent()   # blk2
    y.parent(2)  # blk1


# 向量 Vector
# 在 Taichi 中，向量有两种表述形式：
#   作为临时局部变量，一个由 n 个标量组成的 n 分量向量。
#   作为全局张量(global tensor)的构成元素。比如，一个由 n 分量向量组成的N-维数组构成的全局张量。
# 事实上，向量 是 矩阵 的一个别名，只不过向量的 m = 1 (m 代指列）

# 全局张量中的向量
    ti.Vector.var(n, dt, shape = None, offset = None)
    # 参数:	
    # n – （标量) 向量中的分量数目
    # dt – （数据类型) 分量的数据类型
    # shape – （可选，标量或元组）张量的形状（其中的元素是向量）, 请参阅 张量与矩阵
    # offset – （可选，标量或元组）请参阅 Coordinate offsets

    # 这里我们创建了一个5x4的张量，张量中的元素都是3维的向量:
    # Python 作用域中, ti.var 声明 Tensors of scalars, 而 ti.Vector 声明了由向量构成的张量
    a = ti.Vector.var(3, dt=ti.f32, shape=(5, 4))
# 临时局部变量向量
    ti.Vector([x, y, ...])
    # 参数:	
    # x – （标量）向量的第一个分量
    # y – （标量）向量的第二个分量
    # 例如, 我们可以使用 (2, 3, 4)创建一个三维向量:
    # Taichi 作用域
    a = ti.Vector([2, 3, 4])

# 访问向量的分量
# 全局张量中的向量（即张量下的元素是一个向量）
    a[p, q, ...][i]
    # 参数:	
    # a – （向量张量）
    # p – （标量) 张量的行索引
    # q – （标量) 张量的列索引
    # i – （标量) 向量内分量的索引
    # 这里提取出了向量 a[6, 3] 的第一个分量:
    x = a[6, 3][0]
    # 或者
    vec = a[6, 3]
    x = vec[0]
    # 特别的，对0维张量第一组方括号应该使用 [None]
# 临时局部变量向量
    a[i]
    # 参数:	
    # a – （向量）向量
    # i – 指定访问下标
    # 例如，这里我们提取出了向量 a 的第一个分量:
    x = a[0]
    # 同理，将 a 的第二个分量设置为 4:
    a[1] = 4


a = ti.Vector([3, 4])
# 对于向量，支持
a.norm() # sqrt(3*3 + 4*4 + 0) = 5 等价 ti.sqrt(a.dot(a) + eps)
    # 这里默认参数：a.norm(eps = 0)
    # eps意义：可以通过设置 eps = 1e-5 ，对可微编程中零向量上的梯度值计算进行保护。
a.norm_sqr() # 3*3 + 4*4 = 25 等价 a.dot(a)
a.normalized() # [3 / 5, 4 / 5] 等价 a / a.norm()

# 点积
a = ti.Vector([1, 3])
b = ti.Vector([2, 4])
a.dot(b) # 1*2 + 3*4 = 14

# 叉积（右手系）
a = ti.Vector([1, 2, 3])
b = ti.Vector([4, 5, 6])
c = ti.cross(a, b)
# c = [2*6 - 5*3, 4*3 - 1*6, 1*5 - 4*2] = [-3, 6, -3]
p = ti.Vector([1, 2])
q = ti.Vector([4, 5])
r = ti.cross(a, b)
# r = 1*5 - 4*2 = -3

# 张量积
a = ti.Vector([1, 2])
b = ti.Vector([4, 5, 6])
c = ti.outer_product(a, b) # 注意: c[i, j] = a[i] * b[j]
# c = [[1*4, 1*5, 1*6], [2*4, 2*5, 2*6]]

# 支持维度返回
a.n  # （标量）返回向量 a 的维度
