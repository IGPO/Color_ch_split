import cv2



# Загрузка изображения
img_path = 'кетоны/свет1-вид под углом/full.jpg' 
img_path = 'кетоны/свет0.5-вид сверху/full.jpg' 
image = cv2.imread(img_path)
b, g, r = cv2.split(image)
# Предобработка изображения
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresholded = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]

# Поиск контуров на изображении
contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
can = cv2.Canny(r, 100, 200)
# Отображение контуров на изображении
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

# Отображение изображения с контурами
cv2.imshow('Detected Cards', can)
cv2.waitKey(0)
cv2.destroyAllWindows()