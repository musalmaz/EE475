import cv2

img = cv2.imread('test.jpg')
h, w, c = img.shape
new_size = (w//4, h//4)
resized_bicubic_img = cv2.resize(img, new_size,cv2.INTER_CUBIC)
resized_bilinear_img = cv2.resize(img, new_size,cv2.INTER_LINEAR)
resized_nn_img = cv2.resize(img,new_size,cv2.INTER_NEAREST)
cv2.imwrite("resized_cubicimg.jpg", resized_bicubic_img)
cv2.imwrite("resized_bilinearimg.jpg", resized_bilinear_img)
cv2.imwrite("resized_nnimg.jpg", resized_nn_img)
cv2.imshow("test",img)
cv2.imshow("resized_bicubic",resized_bicubic_img)
cv2.waitKey(0)
