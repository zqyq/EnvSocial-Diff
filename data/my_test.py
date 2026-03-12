from PIL import Image, ImageDraw
import numpy as np
import argparse
import cv2

def mark_coordinates_on_image(image_path, txt_path, output_path, color="red", radius=5):
    """
    处理科学计数法坐标并在图片上标记
    """
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    with open(txt_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue  # 跳过空行和注释

            try:
                # 解析科学计数法
                x_str, y_str = line.split()
                x = float(x_str)
                y = float(y_str)

                # 转换为整数坐标（假设需要像素级精度）
                x = int(round(x))
                y = int(round(y))

                # 验证坐标是否在图片范围内
                if not (0 <= x < img.width and 0 <= y < img.height):
                    print(f"警告：第 {line_num} 行坐标 ({x},{y}) 超出图片范围")
                    continue

                # 绘制标记
                bbox = [(x - radius, y - radius), (x + radius, y + radius)]
                draw.ellipse(bbox, fill=color, outline="black")
                draw.line([(x - radius * 2, y), (x + radius * 2, y)], fill=color, width=2)
                draw.line([(x, y - radius * 2), (x, y + radius * 2)], fill=color, width=2)

            except ValueError as e:
                print(f"错误：第 {line_num} 行格式无效 - {line}")
                print(f"详细信息：{str(e)}")
            except Exception as e:
                print(f"处理第 {line_num} 行时发生未知错误：{str(e)}")

    img.save(output_path)
    print(f"处理完成，结果保存至：{output_path}")


def make_bound(args):
    img = Image.open(args.image)
    draw = ImageDraw.Draw(img)
    width, height = img.size
    radius = args.radius
    xy = [[166, 115], [561, 130], [132, 440], [602, 445]]
    for i in range(len(xy)):
        x = xy[i][0]
        y = height - xy[i][1]
        bbox = [(x - radius, y - radius), (x + radius, y + radius)]
        draw.ellipse(bbox, fill=color, outline="red")
        draw.line([(x - radius * 2, y), (x + radius * 2, y)], fill=color, width=2)
        draw.line([(x, y - radius * 2), (x, y + radius * 2)], fill=color, width=2)
    img.save(args.output)

def test(args):
    length = 13
    width = 12.6
    image = cv2.imread(args.image)
    post1 = np.float32([[166, 115], [561, 130], [132, 440], [602, 445]])
    post2 = np.float32([[0, length],[width, length], [0, 0], [width, 0]])
    M = cv2.getPerspectiveTransform(post1, post2 * 30)
    print(M)
    # # cv2.imshow("image", cv2.warpPerspective(image, M, (width*30,length*30)))
    # warped = cv2.warpPerspective(image, M, (int(width*30), int(length*30)))
    # cv2.imwrite("test.png", warped)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="在图片上标记坐标点")
    parser.add_argument("--image", type=str,  default="UCY/zara01/bg.png" ,help="输入图片路径")
    parser.add_argument("--txt", type=str,  default="UCY/zara01/static.txt",help="坐标文件路径")
    parser.add_argument("--output", type=str, default="obs.png", help="输出图片路径")
    parser.add_argument("--color", type=str, default="blue", help="标记颜色（英文名称或RGB值，如 255,0,0）")
    parser.add_argument("--radius", type=int, default=5, help="标记点半径（像素）")
    args = parser.parse_args()

    # 处理颜色参数
    if "," in args.color:
        try:
            color = tuple(map(int, args.color.split(",")))
        except:
            color = args.color
    else:
        color = args.color

    # 执行标记
    mark_coordinates_on_image(
        image_path=args.image,
        txt_path=args.txt,
        output_path=args.output,
        color=color,
        radius=args.radius
    )

    # make_bound(args)
    # test(args)

