import cv2
import numpy as np
import matplotlib.pyplot as plt

image_lena = cv2.imread('peppers_gray.tif', cv2.IMREAD_GRAYSCALE)
image_lena_color = cv2.imread('PeppersRGB.tif')

# 定义格子的大小
grid_size = (8, 8)
grad_size_sum = grid_size[0] * grid_size[1] * 255
# 获取图像的高度和宽度
height, width = image_lena.shape

line_image_lena = np.ones_like(image_lena) * 255  # 创建白色背景

grid_image = np.copy(image_lena)

# k是灰度值总和和高度max_distance的一个系数关系
k = 0.00015

fig, ax = plt.subplots(figsize=(10.24,10.24))
fig.set_facecolor('white')
# 设置x轴和y轴的范围，左上角为坐标原点
ax.set_xlim(0,1024)
ax.set_ylim(1024,0)
ax.set_aspect('equal')
top_vertices_per_row_lena = []   #定义一个存放每行顶点底点的数组
bottom_vertices_per_row_lena = []
top_vertices_per_row_lena_green = []   #定义一个存放每行顶点底点的数组
bottom_vertices_per_row_lena_green = []
top_vertices_per_column_lena = []
bottom_vertices_per_column_lena = []

# 循环遍历图像，分割成格子并计算每个格子的灰度值总和 以行遍历
for y in range(0, height, grid_size[0]):
    top_row_vertices_lena = []
    bottom_row_vertices_lena = []
    for x in range(0, width, grid_size[1]):
        # 计算当前格子的坐标
        top_left = (x, y)      # 格子的左上角坐标
        bottom_right = (x + grid_size[1], y + grid_size[0])    # 格子的右下角坐标

        # 获取当前格子内的灰度值
        grid_lena = image_lena[y:y + grid_size[0], x:x + grid_size[1]]

        sum_of_grayscale_lena = np.sum(grid_lena)

        # 计算每个格子的中心点坐标
        center_x = x + grid_size[1] // 2
        center_y = y + grid_size[0] // 2

        # 计算每个格子的最左端中心处坐标和最右端中心处坐标
        left_center_x = x
        left_center_y = y + grid_size[0] // 2
        right_center_x = x + grid_size[1]
        right_center_y = y + grid_size[0] // 2

        # 计算每个格子的最高距离max_distance,与每个格子内灰度值sum_of_grayscale有关
        max_distance_lena = (k * (grad_size_sum - sum_of_grayscale_lena))


        # 计算每个格子的最高点和最低点坐标
        max_x = (left_center_x + center_x) // 2
        max_y_lena = y + (grid_size[0] // 2 - max_distance_lena)
        # max_y_lena = y + (grid_size[0] // 2 - max_distance_lena)

        min_x = (right_center_x + center_x) // 2
        min_y_lena = y + grid_size[0] // 2 + max_distance_lena
        # min_y_lena = y + grid_size[0] // 2 + max_distance_lena

        # 每个格子各点分布
        top_vertex_lena = (2 * max_x, 2 * max_y_lena)
        bottom_vertex_lena = (2 * min_x, 2 * min_y_lena)
        left_vertex = (2 * left_center_x, 2 * left_center_y)
        right_vertex = (2 * right_center_x, 2 * right_center_y)
        midpoint = (2 * center_x, 2 * center_y)

        left_center_x_green = left_center_x + grid_size[1] // 4
        left_center_y_green =left_center_y
        right_center_x_green = right_center_x + grid_size[1] // 4
        right_center_y_green = right_center_y


        # # 绘制直线
        # ax.plot([left_vertex_green[0], top_vertex_lena_green[0], bottom_vertex_lena_green[0], right_vertex_green[0]],
        #         [left_vertex_green[1], top_vertex_lena_green[1], bottom_vertex_lena_green[1], right_vertex_green[1]],
        #         color='red', linewidth=1.5)
        # ax.plot([left_vertex[0], top_vertex_lena[0], bottom_vertex_lena[0], right_vertex[0]],
        #         [left_vertex[1], top_vertex_lena[1], bottom_vertex_lena[1], right_vertex[1]],
        #         color='green',linewidth=1.5)
        # # ax.plot([left_vertex_green[0], top_vertex_lena_green[0], bottom_vertex_lena_green[0], right_vertex_green[0]],
        # #         [left_vertex_green[1], top_vertex_lena_green[1], bottom_vertex_lena_green[1], right_vertex_green[1]],
        # #         color='red', linewidth=1.5)

        top_row_vertices_lena.append(top_vertex_lena)
        bottom_row_vertices_lena.append(bottom_vertex_lena)


    top_vertices_per_row_lena.append(top_row_vertices_lena)
    bottom_vertices_per_row_lena.append(bottom_row_vertices_lena)


# 绘制当前行最高点和最低点之间的线条,绘制叶片
for row in top_vertices_per_row_lena:
    for i in range(len(row) - 1):
        ax.plot([row[i][0], row[i + 1][0]], [row[i][1], row[i + 1][1]], color='black', linewidth=0.125,alpha=0.7)
for row in bottom_vertices_per_row_lena:
    for i in range(len(row) - 1):
        ax.plot([row[i][0], row[i + 1][0]], [row[i][1], row[i + 1][1]], color='black', linewidth=0.125,alpha=0.7)



# 在叶片填充黑色
for i in range(len(top_vertices_per_row_lena)-1):
    for j in range(len(bottom_vertices_per_row_lena[i])-1):
        x_values = [
            top_vertices_per_row_lena[i][j][0],
            bottom_vertices_per_row_lena[i][j][0],
            bottom_vertices_per_row_lena[i][j + 1][0],
            top_vertices_per_row_lena[i][j + 1][0],
            top_vertices_per_row_lena[i][j][0]
        ]
        y_values = [
            top_vertices_per_row_lena[i][j][1],
            bottom_vertices_per_row_lena[i][j][1],
            bottom_vertices_per_row_lena[i][j + 1][1],
            top_vertices_per_row_lena[i][j + 1][1],
            top_vertices_per_row_lena[i][j][1]
        ]
        ax.fill(x_values, y_values, color='black', alpha=1.0)

plt.show()

