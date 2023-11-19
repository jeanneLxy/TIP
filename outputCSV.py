import csv
import os

def outputCSV(folder_path, output_csv, data_list):
    files = os.listdir(folder_path)
    with open(output_csv, 'w', newline='') as csvfile:
        # 创建 CSV writer，指定分隔符为逗号
        csv_writer = csv.writer(csvfile, delimiter=',')
        # 写入 CSV 文件的标题（假设为 "image_path" 和 "label"）
        csv_writer.writerow(['image_path', 'label'])

        # 遍历文件夹中的文件并将文件名（去掉后缀）和列表元素写入 CSV 文件
        for file in files:
            # 使用 os.path.splitext 获取文件名和扩展名
            file_name, file_extension = os.path.splitext(file)

            # 检查文件扩展名是否为 '.jpg'
            if file_extension.lower() == '.jpg':
                # 获取列表元素，如果列表为空或长度不够，则使用默认值 0
                label = data_list.pop(0) if data_list else 0

                # 将文件名（去掉后缀）和列表元素写入 CSV 文件的一行
                csv_writer.writerow([file_name, label])

# 设置输出的 CSV 文件路径
output_csv = 'output.csv'
# 设置图像文件夹路径
folder_path = 'test-resized'
# 设置与每个文件对应的标签列表
list_1 = [0, 0, 1, 1, 0, 0, 0, 0]

# 处理图像并写入 CSV 文件
outputCSV(folder_path, output_csv, list_1)

