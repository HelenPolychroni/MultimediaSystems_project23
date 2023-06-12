import cv2
import numpy
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
    
    return 0
    
def subsampling(image):
    
    down_sampled = image[::2,::2]
    #print("new table:\n", down_sampled)
    
    return down_sampled
  
  
'''
Hierarchical Search
'''  
def hierarchical_search(referenceFrame, targetFrame, block_size, k):
    
    # target's frame dimensions
    height, width = targetFrame.shape
    
    print("\nheight ",height)
    print("width ", width)
    
    # macroblock's dimensions
    #b_height, b_width = block_size
    
    
    macroblocks_topl_corner_list = []  # macroblock's top left corner list
    for x in range(0, height, block_size):     # step = b_height
        for y in range(0, width, block_size):  # step = b_width
            
            top_left_corner = (x,y)
            macroblocks_topl_corner_list.append(top_left_corner)  # append to list
            
            #get_searchArea(x, y, referenceFrame, block_size, k)
            
    r_frame_array = []  # reference frame 
    t_frame_array = []  # target frame
    
    r_frame_searchArea_list = []
    
    r_frame_array.append(referenceFrame) # initial reference frame
    t_frame_array.append(targetFrame)    # initial target frame
            
    # assume we do it for the macroblock (64,64) coords 4 the top left corner  
    
    x = 64 # current macroblock's top left corner
    y = 64
    
    level = 1
    size = 1
    while (level < 2):  # leveling up
            
        referenceFrame_area = get_search_area(x, y, referenceFrame, block_size, k) # new reference frame 
        r_frame_searchArea_list.append(referenceFrame_area)
        print("rf\n", referenceFrame)
        
        # Sumsampling
        referenceFrame = subsampling(referenceFrame)  # reference's frame dimensions redused by half
        targetFrame = subsampling(targetFrame)        # target's frame dimensions redused by half
        
        # Converting the numpy array into image
        cv2.imwrite('Image_from_array_rf.png', referenceFrame)
        cv2.imwrite('Image_from_array_tf.png', targetFrame)
        
        # Load the images
        rf_image = cv2.imread('Image_from_array_rf.png')
        tf_image = cv2.imread('Image_from_array_tf.png')
        
        # Apply Gaussian blur filter
        ksize = 3
        sigmaX =1
        referenceFrame = cv2.GaussianBlur(rf_image, (ksize, ksize), sigmaX) # type: ignore
        targetFrame = cv2.GaussianBlur(tf_image, (ksize, ksize), sigmaX) # type: ignore
        
        
        # append new level frames into lists
        r_frame_array.append(referenceFrame)
        t_frame_array.append(targetFrame)
        
        
        block_size = block_size // 2   # macroblock's size redused by half
        k = k // 2                     # search area size redused by half
        print("Search area size (k): ", k)
        
        rows = len(referenceFrame)
        cols = len(referenceFrame[0])
        
        size = rows * cols
        print("Rows: ", rows)
        print("Columns: ", cols)
        print("Image's size: ", size)
        
        level+=1
    
    print("Number of levels: ", level)
    
    
    
    '''
    # round 2
    print("\nCall subsampling function...")
    
    sr_image = subsampling(referenceFrame)
    st_image = subsampling(targetFrame)
    
    get_searchArea(64//2, 64//2, sr_image, block_size//2, k//2)  # till k = 1 or very small
    
    block_size = block_size // 2
    referenceFrame = sr_image
    targetFrame = st_image        
     '''       
            
    
    print("\nend def h_search")
    
    return macroblocks_topl_corner_list,  referenceFrame
 
 
def get_center(x, y, block_size): # usefull for search area (k)
    '''
    Determines center of a block with x, y as top left corner coordinates and block_size as block_size
    -return: x, y coordinates of center of a block
    '''
    block_center = (int(x + block_size/2), int(y + block_size/2))
    return block_center


def get_search_area(x, y, referenceFrame, blockSize, k):
    '''
    Returns image of reference Frame search area
    
    -param x, y: top left coordinate of macroblock in target Frame
    -param referenceFrame: reference Frame
    -param blockSize: size of block in pixels
    -param k: size of search area in pixels
    
    -return: Image of reference Frame search area
    '''
    print("Search area function here!")
    h, w = referenceFrame.shape 
    
    #print("height: ",h)
    #print("widht: ", w)
    center_x, center_y = get_center(x, y, blockSize)

    search_x = max(0, center_x - int(blockSize/2) - k)  # ensure search area is in bounds
    search_y = max(0, center_y - int(blockSize/2) - k)  # and get top left corner of search area

    # slice reference frame within bounds to produce reference search area
    referenceSearch = referenceFrame[search_y:min(search_y + k * 2 + blockSize, h), search_x : min(search_x + k * 2 + blockSize, w)]

    #print (referenceSearch)
    return referenceSearch

def get_new_block_tlc(): # tlc : top left corner
    
    return 0
    
 
  
'''-------------------------------------------------------MAIN-------------------------------------------------------------------------------------------------'''
    
# 1. Collect video frames
#num_of_frames = collect_frames("video1.mp4")

image = get_image("./output/frame_0.jpg")
image = geek.array(image)

# find the rows and cols
rows = len(image)
cols = len(image[0])
print("\nFrame's dimensions")
print("--------------------")
print("Rows: ", rows)
print("Columns: ", cols)

print("\nReference frame is:\n", image)


image1 = get_image("./output/frame_1.jpg")
image1 = geek.array(image1)
print("\nTarget frame is:\n", image1)
   


"__NEW TRY ABOVE:____"    
# ----main program starts here
block_size = 64
k = 32  # search range

l, s_image = hierarchical_search(image, image1, block_size, k)

print(s_image) # new reference frame

#print("Macroblocks dim are:\n", l)
print("\nNumber of macroblocks: ", len(l))


# Converting the numpy array into image
cv2.imwrite('SearchArea.png', s_image) # new reference frame image

# Load the image
s_image = cv2.imread('SearchArea.png')



