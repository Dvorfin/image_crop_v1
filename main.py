from modules import *


def main():
    path = 'scan0001.tif'
    img = cv.imread(path)
    lines = find_countours(img, threshold=133, more_than=9000, less_then=9883)

    for i in range(4):
        lines[i] = np.int0(lines[i])

    angle, lines_coefs = calc_avg_angle(lines)
    rect_p = calc_intersection_points(lines_coefs)

    center = calc_center(rect_p)
    size = calc_rect_size(rect_p)

    img = cv.imread(path)
    rep_points_dict = calc_rep_points(img, rect_p, 40, 60)

    # если нужно повернуть налево
    if rep_points_dict['angle 1'] == 4 and rep_points_dict['angle 4'] == 3:
        angle += 90
        size = (size[1], size[0])
    elif rep_points_dict['angle 2'] == 4 and rep_points_dict['angle 3'] == 3:  # если нужно повернуть направо
        angle -= 90
        size = (size[1], size[0])
    else:
        print('\nCannot find rep points!\n')

    # print(rect_p)

    rect = (center, size, angle)

    print(f'Calc rect: {rect}')

    img = cv.imread(path)
    crop = crop_rot_rect(img, rect)

    # cv.imshow('countours', crop)
    cv.imwrite('crop_res.tif', crop)


res_msg = ''


def iter_main():

    global res_msg

    path = 'C:/Users/Root/Documents/MEGAsync/diplom/scans/10.03.2023/brightness_64/'
    img_names = ['scan0001.tif', 'scan0002.tif', 'scan0004.tif', 'scan0004.tif',
                 'scan0005.tif', 'scan0006.tif', 'scan0007.tif', 'scan0008.tif',
                 'scan0009.tif', 'scan0010.tif']

    res_cnt = 1
    for name in img_names:
        img = cv.imread(path + name)
        lines = find_countours(img, threshold=137, more_than=9000, less_then=9728)

        for i in range(4):
            lines[i] = np.int0(lines[i])

        angle, lines_coefs = calc_avg_angle(lines)
        rect_p = calc_intersection_points(lines_coefs)

        center = calc_center(rect_p)
        size = calc_rect_size(rect_p)

        img = cv.imread(path + name)
        rep_points_dict = calc_rep_points(img, rect_p, 40, 60)

        # если нужно повернуть налево
        if rep_points_dict['angle 1'] == 4 and rep_points_dict['angle 4'] == 3:
            angle += 90
            size = (size[1], size[0])
            res_msg += f'{name} rot + 90\n'

        elif rep_points_dict['angle 2'] == 4 and rep_points_dict['angle 3'] == 3:  # если нужно повернуть направо
            angle -= 90
            size = (size[1], size[0])
            res_msg += f'{name} rot - 90\n'
        else:
            res_msg += f'{name} Cannot find rep points! Change params in func calc_rep_points\n'
            #print('\nCannot find rep points!\n')

        # print(rect_p)

        rect = (center, size, angle)

        print(f'Calc rect: {rect}')

        img = cv.imread(path + name)
        crop = crop_rot_rect(img, rect)

        # cv.imshow('countours', crop)
        cv.imwrite(f'crop_res{res_cnt}.tif', crop)
        res_cnt += 1


if __name__ == '__main__':
    iter_main()

    print(res_msg)