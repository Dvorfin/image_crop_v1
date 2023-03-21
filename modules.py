import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math


show_results = False


def binaryze(image, threshold=220):
    gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    ret, thresh = cv.threshold(gray_img, threshold, 255, cv.THRESH_BINARY)

    if show_results:
        y_range, x_range, _ = image.shape
        cv.namedWindow(f"binaryze {threshold}", cv.WINDOW_NORMAL)  # создаем главное окно
        cv.resizeWindow(f"binaryze {threshold}", int(x_range // 9), int(y_range // 9))  # уменьшаем картинку в 3 раза
        cv.imshow(f"binaryze {threshold}", thresh)
        cv.waitKey(0)
        cv.destroyAllWindows()
    return thresh


def find_countours(img, threshold=133, more_than=0, less_then=2000):
    bin = binaryze(img, threshold)
   #bin = binaryze(img, 140)
    contours0, hierarchy = cv.findContours(image=bin.copy(), mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)

    # массив, куда будут записаны координаты 4-х линий прямоугольника
    lines_of_rectrangle = [[], [], [], []]
    # множество с точками контура
    coords_set = set()

    for cnt in contours0:
        if less_then > cnt.shape[0] > more_than:  # если размер контура найден

            rect = cv.minAreaRect(cnt)  # пытаемся вписать прямоугольник
            box = cv.boxPoints(rect)
            print(f' Opencv rect = {rect}\n')

            print(f'Size of contour: {cnt.shape[0]}')
            for p in cnt:
                for point in p:
                    coords_set.add((point[0], point[1]))

    # поиск точки с минимальным иксом
    A = sorted(coords_set)[0:10]
    min_x = A[0][0]  # выбираем минимальный x
    A1 = []
    for p in A:  # проходимся по всем точкам с минимальными иксами
        if p[0] != min_x:  # если минимальный x изменился
            break
        A1.append(p[1])     # добавляем значения y
    A = (min_x, sum(A1) // len(A1))     # записываем минимальный x и средний y

    # поиск точки с минимальным игриком
    B = sorted(coords_set, key=lambda point: point[1])[0:10]
    #print(B)
    min_y = B[0][1]
    B1 = []
    for p in B:
        if p[1] != min_y:
            break
        B1.append(p[0])
    B = (sum(B1) // len(B1), min_y)

    # поиск точки с максимальным иксом
    C = sorted(coords_set, reverse=True)[0:10]
    max_x = C[0][0]
    C1 = []
    for p in C:
        if p[0] != max_x:
            break
        C1.append(p[1])
    C = (max_x, sum(C1) // len(C1))

    # поиск точки с максимальным игриком
    D = sorted(coords_set, reverse=True, key=lambda point: point[1])[0:10]
    # print(D)
    max_y = D[0][1]
    D1 = []
    for p in D:
        if p[1] != max_y:
            break
        D1.append(p[0])
    D = (sum(D1) // len(D1), max_y)

    print(f'Min x: {A}')
    print(f'Min y: {B}')
    print(f'Max x: {C}')
    print(f'Max y: {D}')


    #------------------------------------------------------------
    #   установка точки A в верхний левый угол
    #------------------------------------------------------------

    height, width, _ = img.shape

    if (A[0] < width // 2) and (A[1] < height // 2):
        angle_points = [A, B, C, D]
    else:
        angle_points = [B, C, D, A]

    # координаты точек угов между которыми будут строиться линии
    angles = [[angle_points[0], angle_points[1]],
              [angle_points[1], angle_points[2]],
              [angle_points[2], angle_points[3]],
              [angle_points[3], angle_points[0]]]

    #------------------------------------------------------------


    # отрисовка 4 точек с их координатами
    cnt = 1
    for point in angle_points:
        cv.circle(img, point, 15, (0, 250, 0), 2)
        cv.putText(img, f'{point}', (point[0] - 40, point[1] + 20),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 250), 1)
        cv.putText(img, f'{cnt}', (point[0] - 40, point[1] + 20),
                   cv.FONT_HERSHEY_SIMPLEX, 10, (0, 0, 250), 10)
        cnt += 1

#--------------------------------------------------------------
    for cnt in contours0:
        if less_then > cnt.shape[0] > more_than:
            # цвета линий
            colors = [(10, 0, 204), (150, 10, 0), (0, 255, 0), (255, 255, 0)]

            # проходимся по всем углам
            for i in range(4):
                point1, point2 = angles[i]  # берем 2 соседние точки
                # вычисялем ширину и высоту прямогуольника, построенного по 2 угловым точкам
                x_r_min, x_r_max = min(point1[0], point2[0]), max(point1[0], point2[0])
                y_r_min, y_r_max = min(point1[1], point2[1]), max(point1[1], point2[1])

                # отрисовка прямогольников
                #cv.rectangle(img, (x_r_min, y_r_min), (x_r_max, y_r_max), (0, 255, 255), 2)

                if i % 2 == 0:  # если горизонтальная линия, то умньшаем прямогольник по ширине
                    x_r_min += 5
                    x_r_max -= 5
                else:           # если вертиакльная линия, то умньшаем прямогольник по высоте
                    y_r_min += 5
                    y_r_max -= 5

                # проходимся по точкам контура
                for p in cnt:
                    for point in p:
                        point = np.int0(point)
                        # если точка контура лежит в пределах прямогольника, то записываем ее
                        if (x_r_min < point[0] < x_r_max) and (y_r_min < point[1] < y_r_max):
                            # print(point)
                            # cv.drawContours(img, cnt, -1, (0, 255, 0), 2, cv.LINE_AA)   # отрсиоввываем найденный контур
                            cv.circle(img, point, 1, colors[i], 2)

                            lines_of_rectrangle[i].append(point)

                # cv.drawContours(img, [box], 0, (255, 0, 100), 2)

    y_range, x_range, _ = img.shape
    cv.namedWindow('countours', cv.WINDOW_NORMAL)  # создаем главное окно
    cv.resizeWindow('countours', int(x_range // 7), int(y_range // 7))  # уменьшаем картинку в 3 раза
    cv.imshow('countours', img)

    if show_results:
        cv.waitKey(0)
    cv.destroyAllWindows()

    cv.imwrite('res.tif', img)

    return lines_of_rectrangle


# функция вычисления k и b коэф-тов прямой
def calc_mnk_k_b(x, y):
    x = np.int0(x)  # преобразование икса в инт
    x = np.array([[x] for x in x])  # преобразование икса в столбец

    col = np.array([[1] for _ in range(len(x))])  # создание второго столбца
    x = np.append(x, col, axis=1)
    xTx = np.matmul(x.T, x)  # произведение матриц (xT * x)
    inv_x = np.linalg.inv(xTx)  # взятие обратной матрицы
    tetta = np.matmul(np.dot(inv_x, x.T), y)  # (xT * x)^T * xT * y

   # print(f'k = {tetta[0]}\nb = {tetta[1]}')

    return tetta[0], tetta[1]


# функция вычисления среднего угла и коэф-тов (k, b)
def calc_avg_angle(lines_data):
    avg_angle = 0
    lines_coef = []     # списко коэффициентов прямых (k, b)

    for i in range(4):  # проходимся по 4 линиям
        x = np.array([])
        y = np.array([])

        for item in lines_data[i]:

            x_t = int(item[0])
            y_t = int(item[1])

            x = np.append(x, y_t)
            y = np.append(y, x_t)

        k, b = calc_mnk_k_b(x, y)   # вычисления коэф-ов k и b
        lines_coef.append([k, b])

        if k > 0:   # если острый или тупой угол
            angle = math.atan(k) + (math.pi / 2)
            angle = math.degrees(angle)
         #   print(f'rot angle: {angle}')

            # пересчет в углы в коордианатах opencv
        else:
            angle = math.pi - math.atan(-1 * k)
            angle = math.degrees(angle)
          #  print(f'rot angle: {angle}')

        if angle < 135:     # пересчет в углы в коордианатах opencv
            avg_angle += (90 - angle)
        else:
            avg_angle += (180 - angle)

        #print()

    print(f'res angle = {avg_angle / 4}')

    return avg_angle / 4, lines_coef


# функция вычисляет точки пересечения 4 линий для наъождения 4 точек углов прямоугольника
# на вход подаются коэф-ты (k и b)
def calc_intersection_points(lines):
    rect_points = []

    for i in range(4):  # проходимся по 4 линиям и ищим точки их пересечения
        k1, b1 = lines[i - 1][0], lines[i - 1][1]
        k2, b2 = lines[i][0], lines[i][1]

        y = (b2 - b1) / (k1 - k2)
        x = k1 * ((b2 - b1) / (k1 - k2)) + b1
       # print(f'x = {x}')
       # print(f'y = {y}')
        rect_points.append([x, y])
        print(f'{i + 1} angle point: {x} | {y}')

        #print()
    return rect_points


# функция вычисляет центр прямоугольника
# на вход подаются коэф-ты (k и b)
def calc_center(rect_points):
    x1, y1 = rect_points[0]
    x2, y2 = rect_points[2]

    k1 = (y2 - y1) / (x2 - x1)
    b1 = (y1 + y2 - k1 * (x1 + x2)) / 2

    x3, y3 = rect_points[1]
    x4, y4 = rect_points[3]

    k2 = (y4 - y3) / (x4 - x3)
    b2 = (y3 + y4 - k2 * (x3 + x4)) / 2

    x = (b2 - b1) / (k1 - k2)
    y = k1 * ((b2 - b1) / (k1 - k2)) + b1

    return x, y


def calc_rect_size(rect_points):
    x1, y1 = rect_points[0]
    x2, y2 = rect_points[2]
    x3, y3 = rect_points[1]
    x4, y4 = rect_points[3]

    # вычисляем ширину верхней и нижней линии и берем среднее
    width_1 = ((x1 - x3) ** 2 + (y1 - y3) ** 2) ** 0.5
    width_2 = ((x2 - x4) ** 2 + (y2 - y4) ** 2) ** 0.5
    width = (width_1 + width_2) / 2
    print(f'Width: {width}')

    # аналогично для высоты
    height_1 = ((x2 - x3) ** 2 + (y2 - y3) ** 2) ** 0.5
    height_2 = ((x1 - x4) ** 2 + (y1 - y4) ** 2) ** 0.5
    height = (height_1 + height_2) / 2
    print(f'Height: {height}')

    return width, height


# на вход подается изображения, 4 точки углов и размеры реперных точек
def calc_rep_points(img, rect_p, more_than=0, less_then=2000):

    res = dict()  # словарь {номер точки: количесво реперных}

    for i in range(4):
        x, y = rect_p[i]
        x_start, x_stop = int(x - 110), int(x + 110)
        y_start, y_stop = int(y - 110), int(y + 110)

        angle_crop = img[y_start:y_stop, x_start:x_stop]
        bin = binaryze(angle_crop, 133)
        # bin = binaryze(img, 140)
        contours0, hierarchy = cv.findContours(image=bin.copy(), mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)

        rep_points_count = 0
        for cnt in contours0:
            if less_then > cnt.shape[0] > more_than:  # если размер контура найден
                cv.drawContours(angle_crop, cnt, -1, (0, 255, 0), 2, cv.LINE_AA)  # отрсиоввываем найденный контур
                rep_points_count += 1
                cv.imshow('rer', angle_crop)

        res.setdefault(f'angle {i + 1}', rep_points_count)

    #print(f'value of points: {val}')
    #cv.imwrite('res.tif', angle_crop)
    print(res)
    return res


 # на вход подать изображение и область (распознанную) по которой обрезать
def crop_rot_rect(img, rect):
    # rect ->  ((центр прямогуольника), (размер прямоугольника), угол наклона)
    center, size, angle = rect[0], rect[1], rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))

    # get row and col num in img
    height, width = img.shape[0], img.shape[1]

    # calculate the rotation matrix
    M = cv.getRotationMatrix2D(center=center, angle=angle, scale=1)  # формирует матрицу поворота
    # rotate the original image
    img_rot = cv.warpAffine(img, M, (width, height))  # вращает изображение

    # now rotated rectangle becomes vertical, and we crop it
    img_crop = cv.getRectSubPix(img_rot, size, center)

    return img_crop
