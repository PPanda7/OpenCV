import cv2

# Загрузка предварительно обученного классификатора для распознавания лиц
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Загрузка изображения
image = cv2.imread('man.jpg')

# Преобразование изображения в черно-белое
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Распознавание лиц на изображении
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

# Отрисовка прямоугольника вокруг обнаруженных лиц
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 4)

# Отображение результата
cv2.imshow('Detected Faces', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
