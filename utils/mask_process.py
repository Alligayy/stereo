
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.linalg import eigh
import colorsys
import math
import cv2

# 求出<=num的切片（等间隔）
def sample_sorted_list(lst:list, num=1000):
    if len(lst) <= num:
        return lst, list(range(len(lst)))
    
    sorted_list = list(range(len(lst)))

    interval = len(sorted_list) // (num - 1) if num > 1 else 1
    sampled_list = sorted_list[::interval]
    if len(sampled_list) > num:
        sampled_list = sampled_list[:num]

    return [lst[i] for i in sampled_list], sampled_list


# 二分查找给定值距离列表最近元素的索引
def find_nearest_index_sorted(lst, value):
    """
    Find the index of the element in a sorted list that is closest to the given value.
    
    Parameters:
    lst (list): A sorted list of numbers.
    value (float): The target number to find the closest element to.
    
    Returns:
    int: The index of the closest element in the list.
    """
    # Binary search to find the closest element
    
    left, right = 0, len(lst) - 1
    while (left <= right):
        mid = (left + right) // 2
        if lst[mid] == value:
            return mid
        elif lst[mid] < value:
            left = mid + 1
        else:
            right = mid - 1

        # if lst[left] == lst[right]:
        #     left, right = mid, mid
        #     break 

    # Determine which side is closer
    if left >= len(lst):
        return right
    elif right < 0:
        return left
    elif abs(lst[right] - value) <= abs(lst[left] - value):
        return right
    else:
        return left
    

def get_best_clip(lst1:list, th:float, max_th=()):

    if not lst1:
        return []
    # lst = sorted(lst1)
    if max_th:
        lst1 = [v for v in lst1 if max_th[0] <= v <= max_th[1]]

    # print(max_th, min(lst1), max(lst1))
    indexed_list = list(enumerate(lst1))
    sorted_indexed_list = sorted(indexed_list, key=lambda x: x[1])
    # print(indexed_list)
    lst = [item[1] for item in sorted_indexed_list]
    sorted_indices = [item[0] for item in sorted_indexed_list]

    max_len, max_clip, max_clip_indices = 0, [], []
    count = 0
    max_range = (-1, -1)
    for i, c in enumerate(lst):
        # print(lst, c, th)
        j = int(find_nearest_index_sorted(lst, c+th))
        clip_lst = lst[int(i):j+1]
        if len(clip_lst) > max_len:
            max_len, max_clip, max_clip_indices = len(clip_lst), clip_lst, sorted_indices[int(i):j+1]
            max_range = (lst[int(i)], lst[int(j)])
    return max_clip_indices, max_range


def find_nearest_mask_point(mask, point):

    mask_points = np.argwhere(mask == 1)
    if point in mask_points.tolist():
        return point
    distances = np.sqrt(np.sum((mask_points - point) ** 2, axis=1))
    min_distance_index = np.argmin(distances)
    return mask_points[min_distance_index].tolist()


def find_nearest_mask_points(mask, point, num_points=3):
    """
    Find the 'num_points' nearest points in a mask matrix to a given point.
    
    :param mask: 2D numpy array representing the mask matrix
    :param point: Tuple (x, y) representing the point coordinates
    :param num_points: Number of nearest points to find
    :return: List of coordinates of the nearest points in the mask
    """

    mask_points = np.argwhere(mask == 1)
    distances = np.sqrt(np.sum((mask_points - point) ** 2, axis=1))
    sorted_indices = np.argsort(distances)
    nearest_indices = sorted_indices[:num_points]

    return mask_points[nearest_indices].tolist()


def fit_ellipsoid(points):
    """
    Fit an ellipsoid to a set of 3D points and return the center and radii.

    Parameters:
    points (np.ndarray): A 2D array of shape (n, 3) containing the 3D points.

    Returns:
    tuple: A tuple containing the center of the ellipsoid (x, y, z) and the radii (r1, r2, r3).
    """
    # 定义椭圆方程的残差函数
    def residuals(params, points):
        a, b, c, d, e, f, g, h, i = params
        residuals = []
        for point in points:
            x, y, z = point
            # 椭圆方程：ax^2 + by^2 + cz^2 + dxy + exz + fyz + gx + hy + iz + 1 = 0
            eq = a*x**2 + b*y**2 + c*z**2 + d*x*y + e*x*z + f*y*z + g*x + h*y + i*z + 1
            residuals.append(eq)
        return residuals

    # 初始参数猜测
    initial_params = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0])

    # 使用最小二乘法拟合椭圆参数
    result = least_squares(residuals, initial_params, args=(points,))

    # 从拟合参数中提取椭圆中心和半径
    a, b, c, d, e, f, g, h, i = result.x

    # 椭圆中心的坐标
    center = np.array([-g / (2 * a), -h / (2 * b), -i / (2 * c)])

    # 构造系数矩阵
    A = np.array([[a, d/2, e/2],
                  [d/2, b, f/2],
                  [e/2, f/2, c]])

    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(A)

    # 特征值对应的就是椭圆的半径的平方
    radii_squared = np.sqrt(np.abs(eigenvalues))

    # 返回椭球中心和三个半径
    return center, np.sqrt(radii_squared)


def min_bounding_box(points):
    """
    计算点云的最小外接长方体。
    参数:
    points: Nx3 的 NumPy 数组，包含点云数据。
    返回:
    center: 3x1 的数组，表示包围盒的中心。
    size: 3x1 的数组，表示包围盒的尺寸（长、宽、高）。
    rotation: 3x3 的数组，表示包围盒的方向。
    """
    # 计算点云的凸包
    hull = ConvexHull(points)
    points = points[hull.vertices]

    # 计算协方差矩阵
    cov = np.cov(points, rowvar=False)
    # 计算协方差矩阵的特征值和特征向量
    eigenvalues, eigenvectors = eigh(cov)

    # 特征向量是按特征值排序的，所以最大的特征值对应的特征向量是长方体的一个轴
    # 由于特征值是按降序排列的，我们取前三个特征向量
    order = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, order]
    center = np.mean(points, axis=0)
    size = np.sqrt(eigenvalues[order])

    # 旋转矩阵（特征向量）需要转置
    rotation = eigenvectors.T

    return center, size, rotation


# def get_decodes(results, names, threeD=None, Q=None, center=False):

def get_decodes(results, names, threeD=None, Q=None, center=False):

    hsv_tuples = [(x / len(names), 1., 1.) for x in range(len(names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    decodes = []
    shape = results.orig_shape
    # 绘制结果
    for i, result in enumerate(results):
        # 解码标注框
        box = results.boxes[i]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(x2, shape[1])
        y2 = min(y2, shape[0])
        class_id = int(box.cls[0])
        class_name = names[class_id]

        mask_flag = False
        # 整理掩码
        try:
            masks = result.masks
            mask_mat = masks.data.detach().cpu().numpy()[0]
            mask_mat = cv2.resize(mask_mat, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
            mask_flag = True
        except:
            mask_mat = np.zeros((shape[0], shape[1]))
            mask_mat[y1:y2, x1:x2] = 1
        
        if (threeD is not None) and (Q is not None):
            if not center:
                # 筛选出深度值对应的掩码区域
                values = threeD[mask_mat > 0]
                z_values = values[:, 2].tolist()

                # 将xyz按照z深度排序，并筛选出<=200个采样点
                # z_values, x_values, y_values = zip(*sorted(zip(z_values, x_values, y_values), key=lambda x: x[0]))
                z_values.sort()
                z_values, indices = sample_sorted_list(z_values, 200)

                #.          .                ....................             ..           .
                # 切出7.5cm内最优切片深度，并筛出其中占比0.3~0.7范围内的值
                # print(z_values)
                sorted_indices, max_range = get_best_clip(z_values, 750, (-float('inf'), 200000))
                sorted_indices = sorted_indices[int(0.3*len(sorted_indices)):int(0.7*len(sorted_indices))]
                z_clip = [z_values[i] for i in sorted_indices]

                z = sum(z_clip) / len(z_clip) if z_clip else 0

                u, v = (x1 + x2) / 2, (y1 + y2) / 2
                x = (u + Q[0][3]) * z / Q[2][3]
                y = (v + Q[1][3]) * z / Q[2][3]
            else:
                u, v = (x1 + x2) // 2, (y1 + y2) // 2
                x, y, z = threeD[v, u]

            width = abs((x2 - x1) * z / Q[2][3])
            height = abs((y2 - y1) * z / Q[2][3])
        else:
            x, y, z, width, height = [None] * 5

        # c = [key for key, value in names.items() if value == class_name][0]
        dict1 = {
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'mask': mask_mat, 
            'class_name':class_name,
            'class_id':class_id,
            'x':x,
            'y':y,
            'z':z,
            'width':width,
            'height':height,
            'mask_flag':mask_flag,
            'color':colors[class_id]
        }
            
        decodes.append(dict1)

    return decodes


def draw_image(img, decodes, scale='cm', digits_num=2, mode='xyz'):

    # 绘制结果
    # imgL_board = imgL.copy()
    imgL = img.copy()
    font_size = 6e-4 * max(imgL.shape[:2])
    box_thickness = max(1, int(2 / 640 * max(imgL.shape[:2])))
    # thickness   = int(2*max((imgL.shape[0] + imgL.shape[1]) // np.mean((imgL.shape[0], imgL.shape[1])), 1))
    
    scale_dict = {
        'mm': 1,
        'cm': 10,
        'm': 1000,
    }
    for i, decode in enumerate(decodes):
        labels = [decode['class_name']]
        color = decode['color']
        cv2.rectangle(imgL, (decode['x1'], decode['y1']), (decode['x2'], decode['y2']), color, box_thickness)

        if decode['x'] is not None:
            x, y, z = decode['x'], decode['y'], decode['z']
            r = math.sqrt(x ** 2 + y ** 2 + z ** 2)

            scale_rate = int(scale_dict[scale])
            r = round(r / scale_rate, digits_num)
            x, y, z = round(x / scale_rate, digits_num), round(y / scale_rate, digits_num), abs(round(z / scale_rate, digits_num))
            
            if 'xyz' in mode:
                label = 'pixel: ({} {} {}){}'.format(x, y, z, scale)
                labels.append(label)
            if 'r' in mode:
                label = 'distance: {} {}}'.format(r, scale)
                labels.append(label)
            if 'wh' in mode:
                w, h = round(decode['width'] / scale_rate, digits_num), round(decode['height'] / scale_rate, digits_num)
                label = '(width {}, height {}){}'.format(w, h, scale)
                labels.append(label)

        # font_size = 0.5
        (_, text_height), _ = cv2.getTextSize(labels[0], cv2.FONT_HERSHEY_SIMPLEX, font_size, -1)
        box_bias = 0 if decode['y1'] - int(text_height * 1.5) * (len(labels) - 1) >= 0 else text_height * (len(labels)+1)

        # if decode['y1'] - int(text_height * 1.5) * (len(labels) - 1) >= 0:

        # 绘制文字底框
        for j, label in enumerate(labels):
            
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, -1)
            bias = int(text_height * 1.5)
            cv2.rectangle(imgL, (decode['x1'], decode['y1'] - bias * (len(labels) - j) + box_bias), (decode['x1'] + text_width, decode['y1'] + box_bias), color, -1)
        
        # 绘制文字
        for j, label in enumerate(labels):
            
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, -1)
            bias = int(text_height * 1.5) 
            font_thickness = max(1, int(font_size / 0.5))

            color_text = (255, 255, 255) if sum(color) / 3 < 128 else (0, 0, 0)

            cv2.putText(imgL, label, (decode['x1'], decode['y1'] - bias * int(len(labels) - j - 1) - int(text_height * 0.25) + box_bias), cv2.FONT_HERSHEY_SIMPLEX, font_size, color_text, font_thickness, cv2.LINE_AA)

        
        if decode['mask_flag']:
            overlay = np.zeros_like(imgL)
            overlay[decode['mask'] > 0] = list(color)

            imgL = cv2.addWeighted(imgL, 1, overlay, 0.4, 0)
            
            color_contour = tuple([c * 0.5 for c in color] + [0.5])
            contours, _ = cv2.findContours(decode['mask'].astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            imgL = cv2.drawContours(imgL, contours, -1, color_contour, thickness=2)



    return imgL


