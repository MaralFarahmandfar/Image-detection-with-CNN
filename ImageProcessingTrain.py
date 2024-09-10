import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.src.utils import to_categorical
from keras.src.models import Sequential
from keras.src.layers import InputLayer,Conv2D,Dense,MaxPooling2D,Flatten
from keras.src.metrics import CategoricalAccuracy
import matplotlib.pyplot as plt

#بارگذاری اطلاعات
path="E:\\.vscode\\imageProcessing\\imagesTrain"
images=[]
image_name = os.listdir(path)

for img in image_name:
    image_path = os.path.join(path, img)
    image=Image.open(image_path)
    image=image.resize((300,300))
    img_array=np.array(image)/255
    images.append(img_array)

image_array=np.array(images)
labels=np.array([0]*14+[1]*14)

#جداسازی داده آموزش و تست
X_train, X_test, y_train, y_test = train_test_split(image_array, labels, test_size=0.4, stratify=labels, random_state=42)

#کلاس بندی
class_name={
    0:'mobile' 
    ,   1: 'pen'
}

#One-Hot Encoding
#هر کلاس به یک آرایه باینری تبدیل و فقط یک عنصرش مقدار 1 دارد
y_train=to_categorical(y_train,num_classes=2)
y_test=to_categorical(y_test,num_classes=2)

#CNNساخت مدل 
model=Sequential()
model.add(InputLayer((300,300,3)))
model.add(Conv2D(filters=16,kernel_size=(3,3),strides=1,activation='relu',padding='same'))
model.add(MaxPooling2D())
model.add(Conv2D(filters=32,kernel_size=(3,3),strides=1,activation='relu',padding='same'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(2,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=[CategoricalAccuracy()])
#آموزش مدل
model.fit(X_train,y_train,epochs=20,batch_size=32,validation_split=0.2)
#ذخیره مدل
model.save("E:\\.vscode\\imageProcessing\\ConvModel.h5")

#پیش بینی
y_pred=model.predict(X_test)

#نمایش
index=7
plt.imshow(X_test[index])
plt.title(f"True: {class_name[np.argmax(y_test[index])]}, Pred: {class_name[np.argmax(y_pred[index])]}")
plt.show()