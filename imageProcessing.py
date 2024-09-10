import os
from PIL import Image
import numpy as np
from keras._tf_keras.keras.models import load_model
from keras.src.utils import to_categorical
import matplotlib.pyplot as plt

#عکس راهنما
guide_image = Image.open("guide.png")
plt.imshow(guide_image)
plt.axis('off')
plt.show()

#بارگذاری اطلاعات
path=input("Enter the path of the photos: ")
images=[]
image_name = os.listdir(path)

for img in image_name:
    image_path = os.path.join(path, img)
    image=Image.open(image_path)
    image=image.resize((300,300))
    img_array=np.array(image)/255
    images.append(img_array)

image_array=np.array(images)

n=int(input("Number of mobile photos: "))
m=int(input("Number of pen photos: "))
labels=np.array([0]*n +[1]*m)

X_test=image_array
y_test=labels

#کلاس بندی
class_name={
    0:'mobile' 
    ,   1: 'pen'
}

#One-Hot Encoding
y_test=to_categorical(y_test,num_classes=2)

#بارگذاری مدل
loaded_model=load_model("ConvModel.h5")

#پیش بینی
y_pred=loaded_model.predict(X_test)

def show():
    #نمایش
    index=int(input("Enter number of photos: "))-1
    plt.imshow(X_test[index])
    plt.title(f"True: {class_name[np.argmax(y_test[index])]}, Pred: {class_name[np.argmax(y_pred[index])]}")
    plt.show()

# اجرای برنامه در یک حلقه
while True:
    show()
    repeat = input("Do you want to test another photo? (yes/no): ").strip().lower()
    if repeat != 'yes':
        break


