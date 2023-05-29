import math
import cv2
import numpy
import numpy as np
import numpy as geek
from PIL import Image

'''Function to extract pixels from an image'''
def get_image(image_path):
    """Get a numpy array of an image so that one can access values[x][y]."""
    image = Image.open(image_path, "r")
    width, height = image.size
    pixel_values = list(image.getdata())
    #print("Image mode: ",image.mode,"\n")
    if image.mode == "RGB":
        channels = 3
    elif image.mode == "L": # greyscale image
        channels = 1
    else:
        print("Unknown mode: %s" % image.mode)
        return None
    pixel_values = numpy.array(pixel_values).reshape((width, height, channels))
    pixel_values_ = pixel_values.reshape(pixel_values.shape[0], -1)
    return pixel_values_


'''Function to collect video frames'''
def collect_frames(video):
    
    capture = cv2.VideoCapture(video)
    frameNr = 0
    
    while (True):
        success, frame = capture.read()
        
        if success:
            #print("yesss")
            cv2.imwrite(f'./output/frame_{frameNr}.jpg', frame)
            frameNr = frameNr+1
        
        else:
            #print("noo")
            break
    
    capture.release()
    return frameNr

def someName(referenceFrame, targetFrame):
    
    # divide targetFrame into macroblocks 64x64
    
    return 0


def macroblocks_coord(img):
    
    shape = img.shape # rows and cols
    block = 64
  
    x_len = shape[0]//block  # num of rows of macroblocks 
    y_len = shape[1]//block  # num of columns of macroblocks
    
    print(x_len, y_len)
    
    x_indices = [i for i in range(0, shape[0]+1, block)]
    y_indices = [i for i in range(0, shape[1]+1, block)]

    shapes = list(zip(x_indices, y_indices))
    
    #c = []
    print("rows: ",shape[0])
    print("columns: ",shape[1])
    
    for i in range(0,shape[0],64):      # step = 64
        for j in range(0,shape[1],64):  # step = 64
            a = (i,j)
            #print("a is: ",a)
            #b = (i+64,j)
            shapes.append(a)  # type: ignore
            #shapes.append(b)  
    
    for i in range(64,shape[0],64):
        for j in range(0,shape[1],64):
           
            b = (i,j)
            shapes.append(b)
            
    shapes = list(dict.fromkeys(shapes))
    
    return shapes
    
    
'''
Encoder Function
'''    
def encoder(rFrame, tFrame):
    '''
    Has to:
    
    '''
    errorImage = tFrame - rFrame # encode this image with JPEG encoding
    motionVectors = []           # encode using Huffman encoding
    
    return errorImage, motionVectors


'''
Decoder Function
'''
def decoder(errorImage, motionVectors):
    '''
    Has to:
    1. Reconstruct Frame1 =  Reference Frame
    2. Reconstruct Frame2 = Target Frame by:
       - Encoding motion vectors (Huffman encoding)
       - Encoding error image  given by the encoder function!
    '''
    
            
 

'''-------------------------------------------------------MAIN-------------------------------------------------------------------------------------------------'''
    
# 1. Collect video frames
#num_of_frames = collect_frames("video1.mp4")


# 2. Find the sequence of error images

# find the rows and cols
image = get_image("./output/frame_0.jpg")
image = geek.array(image)
#print("frame0:\n",image)
rows = len(image)
cols = len(image[0])


image = get_image(f"./output/frame_{0}.jpg")  # first frame
   
print("\nCall macroblocks func...")
l = macroblocks_coord(image)
print(l) 
print("\nmacroblocks: ",len(l)) 
   
print("Telos diadikasias")  
    
#print("Num of frames is: ",num_of_frames)


