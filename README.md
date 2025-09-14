# Color_ch_split
Описание алгоритма цветового сопоставления

Данный алгоритм предназначен для определения наиболее близкого по среднему цвету эталонного изображения к заданному тестовому изображению. Он сравнивает изображения как в цветовом пространстве BGR (используя Евклидово расстояние), так и в цветовом пространстве Lab (используя метрику CIEDE2000).
Шаги алгоритма:

    Загрузка изображений:

        Загружается тестовое изображение (test.jpg).

        Загружается набор эталонных изображений (1.jpg - 5.jpg).

    Вычисление среднего цвета BGR:

        Для каждого изображения (тестового и эталонных) рассчитывается средний цвет по каждому из каналов B, G, R. 
	Результатом является одномерный массив из трех значений [B, G, R].

    Вычисление среднего цвета Lab:

        Каждое изображение конвертируется из BGR в цветовое пространство Lab.

        Для каждого изображения рассчитывается средний цвет по каждому из каналов L, a, b.

        Значения L, a, b нормализуются: L приводится к диапазону [0, 100], а и b к диапазону [-128, 127] (стандартные диапазоны для CIEDE2000). 
	Результатом является одномерный массив из трех значений [L, a, b].

    Расчет расстояний в BGR:

        Для среднего цвета тестового изображения вычисляется Евклидово расстояние до среднего цвета каждого эталонного изображения.

        Наименьшее расстояние определяет наиболее близкий эталон в BGR.

    Расчет расстояний в Lab:

        Для среднего цвета тестового изображения вычисляется цветовое расстояние CIEDE2000 до среднего цвета каждого эталонного изображения.

        Наименьшее расстояние определяет наиболее близкий эталон в Lab.

    Вывод результатов:

        Отображаются все рассчитанные расстояния для обеих метрик (BGR и Lab) для каждого эталонного изображения.

        Указывается индекс наиболее близкого эталонного изображения для каждой метрики.

        Выводятся средние цвета тестового изображения и ближайших эталонных изображений для каждой метрики.

Используемые библиотеки:

    cv2 (OpenCV): Для загрузки изображений и преобразования цветовых пространств.

    numpy: Для работы с массивами и математических операций (например, расчет среднего).

    skimage.color: Для вычисления расстояния CIEDE2000 (deltaE_ciede2000).

    scipy.spatial.distance: Для вычисления Евклидова расстояния (distance.euclidean).

Color Matching Algorithm Description

This algorithm is designed to identify the reference image that is most similar in average color to a given test image. 
It compares images in both BGR color space (using Euclidean distance) and Lab color space (using the CIEDE2000 metric).
Algorithm Steps:

    Image Loading:

        The test image (test.jpg) is loaded.

        A set of reference images (1.jpg - 5.jpg) is loaded.

    Calculate Mean BGR Color:

        For each image (test and reference), the mean color is calculated for each B, G, R channel. The result is a 3-element array [B, G, R].

    Calculate Mean Lab Color:

        Each image is converted from BGR to the Lab color space.

        For each image, the mean color is calculated for each L, a, b channel.

        The L, a, b values are normalized: L to the [0, 100] range, and a and b to the [-128, 127] range (standard ranges for CIEDE2000 calculations). 
	The result is a 3-element array [L, a, b].

    Calculate BGR Distances:

        The Euclidean distance is computed between the mean BGR color of the test image and the mean BGR color of each reference image.

        The smallest distance identifies the closest reference in BGR.

    Calculate Lab Distances:

        The CIEDE2000 color difference is computed between the mean Lab color of the test image and the mean Lab color of each reference image.

        The smallest distance identifies the closest reference in Lab.

    Output Results:

        All calculated distances for both metrics (BGR and Lab) are displayed for each reference image.

        The index of the closest reference image for each metric is reported.

        The mean colors of the test image and the closest reference images are output for each metric.

Libraries Used:

    cv2 (OpenCV): For loading images and converting color spaces.

    numpy: For array manipulation and mathematical operations (e.g., mean calculation).

    skimage.color: For calculating the CIEDE2000 distance (deltaE_ciede2000).

    scipy.spatial.distance: For calculating the Euclidean distance (distance.euclidean).