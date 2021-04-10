# 张量元素也可以是矩阵
# 假设你有一个名为 A 的 128 x 64 张量，每个元素都包含一个 3 x 2 矩阵。 要分配 3 x 2 矩阵的 128 x 64 张量，请使用声明：
# A = ti.Matrix(3, 2, dt=ti.f32, shape=(128, 64))

# 由于性能原因，矩阵运算将被展开，因此建议仅使用小型矩阵
# 由于展开机制，在大型矩阵（例如 32x128 ）上进行操作会导致很长的编译时间和较低的性能
#
# 如果你的矩阵有个维度很大（比如 64），最好定义一个大小为 64 的张量。比如，声明一个
# ti.Matrix(64, 32, dt=ti.f32, shape=(3, 2))
# 是不合理的，可以试着用
# ti.Matrix(3, 2, dt=ti.f32, shape=(64, 32))
# 代替——始终把大的维度放在张量里

# 矩阵 Matrix

# ti.Matrix 只适用于小矩阵（如3x3）。如果要使用 64x64 的矩阵，你可以用标量构成的二维张量。
# ti.Vector 和 ti.Matrix 相同，只不过它只有一列。
# 注意区分逐元素的乘法 * 和矩阵乘法 @
# ti.Vector(n, dt=ti.f32) 或 ti.Matrix(n, m, dt=ti.f32) 用来创建向量/矩阵构成的张量。

A.transpose()
R, S = ti.polar_decompose(A, ti.f32)
U, sigma, V = ti.svd(A, ti.f32) （其中 sigma 是一个 3x3 矩阵）

# Taichi中的矩阵有两种形式：
    # 作为临时局部变量。一个由 n*m 个标量构成的 n×m 阶矩阵。
    # 作为全局张量的一个成员。在这种情况下，张量是一个由 n×m 阶矩阵构成的N-维的数组。

# 作为全局张量的矩阵
    ti.Matrix.var(n, m, dt, shape = None, offset = None)
    # 参数:	
    # n – （标量）矩阵的行数
    # m – （标量）矩阵的列数
    # dt – （数据类型）元素的数据类型
    # shape – （可选，标量或元组）向量张量的形状，见 张量与矩阵
    # offset – （可选，标量或元组）请参见 Coordinate offsets
    # 例如， 以下创建了一个由 3x3 矩阵构成的 5x4 的张量：
    # Python-scope中，ti.var 声明了 Tensors of scalars , ti.Matrix 声明了由矩阵组成的张量。
    a = ti.Matrix.var(3, 3, dt=ti.f32, shape=(5, 4))
# 作为一个临时的本地变量
    # ti.Matrix([[x, y, ...][, z, w, ...], ...])
    # 参数:	
    # x – （标量）第一行第一个元素
    # y – （标量）第一行第二个元素
    # z – （标量）第二行第一个元素
    # w – （标量）第二行第二个元素
    # 例如，下述将创建一个 2x3 的矩阵，第一行中的分量为 (2, 3, 4) ，第二行的为 (5, 6, 7) 中：
    # Taichi-scope
    a = ti.Matrix([[2, 3, 4], [5, 6, 7]])

    # 也可以直接通过Vector拼凑向量
    # Taichi 作用域
    v0 = ti.Vector([1.0, 2.0, 3.0])
    v1 = ti.Vector([4.0, 5.0, 6.0])
    v2 = ti.Vector([7.0, 8.0, 9.0])

    # 指定行中的数据
    a = ti.Matrix.rows([v0, v1, v2])

    # 指定列中的数据
    a = ti.Matrix.cols([v0, v1, v2])

    # 可以用列表代替参数中的向量
    a = ti.Matrix.rows([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

# 元素访问
# 作为全局的由向量构成的张量
    a[p, q, ...][i, j]
    # 参数:	
    # a – （矩阵构成的张量）张量名
    # p – （标量）张量的第一维的索引
    # q – （标量）张量的第二维的索引
    # i – （标量）矩阵的行索引
    # j – （标量）矩阵的列索引
    # 以下代码用以访问矩阵 a[6,3] 的第一个元素：
    x = a[6, 3][0, 0]
    # 或者
    mat = a[6, 3]
    x = mat[0, 0]
    # 同样的，对于0维的矩阵形式的张量，第一对方括号中的索引应该为 [None]

# 作为一个临时的本地变量
    a[i, j]
    # 参数:	
    # a – （矩阵）该矩阵本身
    # i – （标量）矩阵的行索引
    # j – （标量）矩阵的列索引
    # 比如，访问矩阵 a 第0行第1列的元素：
    x = a[0, 1]
    # 将 a 第1行第3列的元素设置为4：
    a[1, 3] = 4

# 矩阵操作
# 求转置
    a = ti.Matrix([[2, 3], [4, 5]])
    b = a.transpose()
    # 现在 b = ti.Matrix([[2, 4], [3, 5]])
# 求迹
    a.trace() # 返回值可以计算为 a[0, 0] + a[1, 1] + ...
# 目前用于计算 行列式和逆 的矩阵大小必须为 1x1、2x2、3x3 或 4x4；且仅在 Taichi 作用域内有效
# 求行列式
    a.determinant()
# 求逆
    a.inverse()


# 结构化节点(SNodes)
# 在编写计算部分的代码之后，用户需要设定内部数据结构的层次。包括微观和宏观两部分
# 宏观上设定层级数据结构组件之间的嵌套关系以及表示稀疏性的方式
# 微观上，描述数据如何分组(例如，SOA 或 AOS)
# Taichi 提供了 结构节点 (SNodes) 以满足不同层级数据结构构建时的需求。其结构和语义具体如下所示：
    # 稠密集合(dense)：固定长度的连续数组。
    # 位掩码集合(bitmasked)：类似于稠密集合，但实现了通过掩码保持数据的稀疏信息。比如为稠密集合的元素分配掩码来记录稀疏信息。
    # 指针集合(pointer)：存储指针而不是整个结构，以节省内存和保持稀疏性。
    # 动态集合(dynamic)：可变长度数组，具有预定义的最大长度。它有着像 C++ 中的 std::vector 或者是 Python 中的 list 这样的功能，可以用来维护包含在一个块(block)中的对象（例如粒子）
# 其中，ti.root 是层级数据结构的根结点

snode.place(x, ...)
# 参数:	
# snode – (结构节点) 放置(place)操作的目标
# x – (张量) 要放置的张量对象
# 返回:	
# (结构节点) snode 对象
# 以下示例代码放置了 x 和 y 两个零维张量:
x = ti.var(dt=ti.i32)
y = ti.var(dt=ti.f32)
ti.root.place(x, y)
assert x.snode() == y.snode()

tensor.shape() # 返回:	(整数元组) 张量的形状
# 相当于 tensor.snode().shape
ti.root.dense(ti.ijk, (3, 5, 4)).place(x)
x.shape # 返回 (3, 5, 4)

tensor.snode() # 返回: （结构节点） tensor 所在的结构节点
x = ti.var(dt=ti.i32)
y = ti.var(dt=ti.f32)
ti.root.place(x, y)
x.snode()

snode.shape() # （元组）张量在指定轴上的尺寸
blk1 = ti.root
blk2 = blk1.dense(ti.i,  3)
blk3 = blk2.dense(ti.jk, (5, 2))
blk4 = blk3.dense(ti.k,  2)
blk1.shape  # ()
blk2.shape  # (3, )
blk3.shape  # (3, 5, 2)
blk4.shape  # (3, 5, 4)

snode.parent(n = 1) # （结构节点） snode 的父类节点  n：代表向上索引父节点的步数
blk1 = ti.root.dense(ti.i, 8)
blk2 = blk1.dense(ti.j, 4)
blk3 = blk2.bitmasked(ti.k, 6)
blk1.parent()  # ti.root
blk2.parent()  # blk1
blk3.parent()  # blk2
blk3.parent(1) # blk2
blk3.parent(2) # blk1
blk3.parent(3) # ti.root
blk3.parent(4) # None

# 不同类型的节点
snode.dense(indices, shape)
    # 参数:	
    # snode – （结构节点） 父节点，返回的子节点从该节点派生
    # indices – （索引）用于子节点上的索引
    # shape – （标量或元组）指定向量张量(tensor of vector)的形状
    # 返回:	
    # （结构节点）派生出来的子节点
    # 以下示例代码放置了尺寸为 3 的一维张量：
    x = ti.var(dt=ti.i32)
    ti.root.dense(ti.i, 3).place(x)
    # 以下示例代码放置了尺寸为 (3,4) 的二维张量：
    x = ti.var(dt=ti.i32)
    ti.root.dense(ti.ij, (3, 4)).place(x)
    # 注意：如果给定的 shape 是一个标量，却又对应了多个索引，那么 shape 将自动扩充直至和索引数量相等。例如,
    snode.dense(ti.ijk, 3)
    # 相当于
    snode.dense(ti.ijk, (3, 3, 3))

snode.dynamic(index, size, chunk_size = None)
    # 参数:	
    # snode – （结构节点） 父节点，返回的子节点从该节点派生
    # index – （索引） 动态集合节点(dynamic node)的索引
    # size – （标量）描述该动态集合节点的最大尺寸
    # chunk_size – （可选标量）描述动态内存分配时块(chunk)中存储的元素数目
    # 返回:	
    # （结构节点）派生出来的子节点
    # 动态集合 节点就像 C++ 中的 std::vector 或者是 Python 中的 list
    # Taichi 具有的动态内存分配系统可以实现自由的分配内存
    # 以下示例代码放置了最大尺寸为 16 的一维动态张量：
    ti.root.dynamic(ti.i, 16).place(x)

# 动态集合节点的使用
ti.length(snode, indices)
    # 参数:	
    # snode – （动态集合节点）
    # indices – (标量或元组中标量) 动态集合 节点的索引
    # 返回:	
    # （int32）当前动态集合节点的尺寸
ti.append(snode, indices, val)
    # 参数:	
    # snode – （动态集合节点）
    # indices – (标量或元组中标量) 动态集合 节点的索引
    # val – （取决于结构节点的数据类型）想要储存的值
    # 返回:	
    # (int32) 进行附加操作之前的动态节点尺寸
# 使用上述函数，就能实现通过 索引(indices) 将 常量(val) 插入到 动态集合 节点中。