from PIL import Image
import os
import face_recognition

PIC_PATH = r'D:\diplom\data1\r'
width = 150
height = 150

def pictures():
    for pt in os.listdir(PIC_PATH):
       #print(pt)
        for img_path in os.listdir(PIC_PATH+"\\"+pt):
            image = face_recognition.load_image_file(PIC_PATH+"\\"+pt+"\\"+img_path)
            face_locations = face_recognition.face_locations(image)
            
            for face_location in face_locations:

                top, right, bottom, left = face_location
                print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
                #print(PIC_PATH+"\\"+pt+"\\"+img_path)
                face_image = image[top:bottom, left:right]
                image = Image.fromarray(face_image)
                

                image.save(PIC_PATH+"\\"+pt+"\\"+img_path)
            
            image = Image.open(PIC_PATH+"\\"+pt+"\\"+img_path)
            resized_img=image.resize((width, height), Image.ANTIALIAS)
            resized_img.save(PIC_PATH+"\\"+pt+"\\"+img_path)

pictures()

    