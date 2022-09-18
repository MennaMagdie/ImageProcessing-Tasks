import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt


# TASK 1: Image Concatenation
def Task1():
    width = 250
    height = 250
    dim = (width,height)

    dog = cv.resize( cv.imread("dog.jpg") , dim)
    cow = cv.resize( cv.imread("student.jpg") , dim)
    duck = cv.resize( cv.imread("duck.jpg") , dim)
    monkey = cv.resize( cv.imread("monkey.jpg") , dim)

    h1 = cv.hconcat([dog,cow])
    h2 = cv.hconcat([duck,monkey])
    Animaux = cv.vconcat([h1,h2])
    cv.imshow('Animaux',Animaux)
    cv.waitKey(0)


# TASK 2: Feature Matching (Brute Force Matcher, SIFT)
def Task2():
    img1 = cv.imread('book.jpg',cv.IMREAD_GRAYSCALE)       
    img2 = cv.imread('allBooks2.jpg',cv.IMREAD_GRAYSCALE) 

    sift = cv.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
        
    img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3),plt.show()


# Task 3: Thresholding TO_ZERO Implementation using numpy
def Task3():
    image = cv.imread("dog.jpg",cv.IMREAD_GRAYSCALE )
    #print(image)
    thresh = 120
    #ret,thresh1 = cv.threshold(image,120,255,cv.THRESH_TOZERO)
    #print(thresh1)

    image[image < thresh] = 0
    #print(image)

    cv.imshow('Dog', image)
    cv.waitKey(0)


# Task 4: High/Low Pass Filters
    """
    High Pass Filters : Preserve high frequency, attenuate the low frequency, used for sharpening an image, helps in removal of noises
    Examples: Sharpening - Edge Detection

    Low Pass Filters : Preserve low frequency, attenuate the high frequency, used for smoothing an image, helps in removal of aliasing effect
    Examples: Blurring

    """


# Task 5: Sharpening an Image
def Task5():
    img = cv.imread('dog.jpg')
    blurred = cv.blur(img , (20,20))

    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
                       
    sharpened = cv.filter2D(img, -1, kernel)
    
    cv.imshow('Differences',cv.hconcat([blurred, sharpened]))
    cv.waitKey(0)


# Task 6: Canny Edge Detection
def Task6():
    before = cv.imread("Canny.jpg" , cv.IMREAD_GRAYSCALE)
  
    t_lower = 60  # Lower Threshold
    t_upper = 250  # Upper threshold
    after = cv.Canny(before, t_lower, t_upper)

    diff = cv.hconcat([before,after])
    cv.imshow('Edge Detection', diff)
    cv.waitKey(0)



Task1()
Task2()
Task3()
Task5()
Task6()