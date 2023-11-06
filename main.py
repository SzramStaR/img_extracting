from skimage import data, color, measure,filters,io
import matplotlib.pyplot as plt
import skimage.io
import numpy as np

image = data.coins()  # Load the "coins" image

def filterImage(img):
    filtered_img = filters.gaussian(img)
    return filtered_img

def findContours(img):
    return measure.find_contours(img)

images = {
    'planes/samolot02.jpg',
    'planes/samolot17.jpg',
    'planes/samolot08.jpg',
    'planes/samolot09.jpg',
    'planes/samolot10.jpg',
    'planes/samolot11.jpg',
}

plt.figure(figsize=(10, 4))

for i,img in enumerate(images,start=1):
    plt.subplot(2,3,i)
    loaded_img = skimage.io.imread(img, as_gray=True)
    filtered_img = filterImage(loaded_img)
    contours = findContours(filtered_img)
    black_img = np.zeros((loaded_img.shape[0],loaded_img.shape[1])) #shape[0] for height , shape[1] for width
    plt.imshow(black_img, cmap='gray')
    
    #remove axis
    plt.axis('off')

    for contour in contours:
        plt.plot(contour[:, 1], contour[:, 0], linewidth=2, c='white')

plt.subplots_adjust(wspace=0,hspace=0)
plt.savefig('contours.pdf')
plt.show()
