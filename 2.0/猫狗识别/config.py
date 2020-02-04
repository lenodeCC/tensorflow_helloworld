import os

# 数据所在文件夹
baseDir = "./data/cats_and_dogs"
trainDir = os.path.join(baseDir, "train")
validationDir = os.path.join(baseDir, "validation")

# 训练集
trainCategoryDirs = {
    "cats": os.path.join(trainDir, 'cats'),
    "dogs": os.path.join(trainDir, 'dogs')
}

# 验证集
validationCategoryDirs = {
    "cats": os.path.join(validationDir, 'cats'),
    "dogs": os.path.join(validationDir, 'dogs')
}
