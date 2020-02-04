# 预处理图像预处理
import tensorflow as tf
import config
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator

# 归一化
trainDatagen = ImageDataGenerator(rescale=1./255)
testDatagen = ImageDataGenerator(rescale=1./255)

# 申明训练和测试集的生成器
trainGenerator = trainDatagen.flow_from_directory(
    config.trainDir,  # 文件夹路径
    target_size=(224, 224),  # 指定resize成的大小
    batch_size=20,
    # 如果one-hot就是categorical，二分类用binary就可以
    class_mode='binary')

validationGenerator = testDatagen.flow_from_directory(
    config.validationDir,
    target_size=(224, 224),
    batch_size=20,
    class_mode='binary')
