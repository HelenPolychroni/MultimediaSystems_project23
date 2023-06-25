import cv2
import numpy
import numpy as geek
import numpy as np
from scipy import signal
from PIL import Image
import matplotlib.pyplot as mp
import imageio

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


'''A Huffman Tree Node'''
class Node:
    def __init__(self, prob, symbol, left = None, right = None):
        # probability of symbol
        self.prob = prob
        
        # symbol
        self.symbol = symbol
        
        # left node
        self.left = left
        
        # right node
        self.right = right
        
        # tree direction (0/1)
        self.code = ''
        
        
'''Function to print the codes of symbols by travelling Huffman Tree'''
codes = dict()

def calculate_codes(node, val = ''):
    # huffman code for current node
    newVal = val + str(node.code)
    
    if(node.left):
        calculate_codes(node.left, newVal)
    if(node.right):
        calculate_codes(node.right, newVal)
    
    if(not node.left and not node.right):
        codes[node.symbol] = newVal
        
    return codes
    
    
'''Function to calculate the probabilities of symbols in given data'''
def calculate_probability(data):
    
    symbols = dict()
    for element in data:
        if symbols.get(element) == None:
            symbols[element] = 1
        else: 
            symbols[element] += 1 
                
    return symbols

    
'''Function to obtain the encoded output''' 
def output_encoded(data, coding):
   
    encoding_output = []
    for d in data:
        encoding_output.append(coding[d])
    
    #string = ''.join([str(item) for item in encoding_output])
    return encoding_output  # list
    
'''Huffman Encoder'''
def Huffman_encoding(data):
    
    symbol_with_probs = calculate_probability(data)
    symbols = symbol_with_probs.keys()
    probabilities = symbol_with_probs.values()
    
    #print("\nSymbols:\n", symbols)
    #print("\nProbabilities:\n", probabilities)
    
    nodes = []
    
    # converting symbols and probabilities into huffman tree nodes
    for symbol in symbols:
        nodes.append(Node(symbol_with_probs.get(symbol), symbol))
        
    while len(nodes) > 1:
        # sort all the nodes in ascending order based on their probability
        nodes = sorted(nodes, key=lambda x: x.prob)
        
        # pick 2 smallest nodes
        right = nodes[0]
        left = nodes[1]
        
        left.code = 0
        right.code = 1
        
        # combine the 2 smallest nodes to create new node
        newNode = Node(left.prob + right.prob, left.symbol + right.symbol, left, right)
        
        nodes.remove(left)
        nodes.remove(right)
        nodes.append(newNode)
    
    
    huffman_encoding = calculate_codes(nodes[0])
    print("\nSymbols with codes:\n", huffman_encoding)
    
    encoded_output = output_encoded(data, huffman_encoding)
    #print("\nEncoded output:", encoded_output)
    
    return encoded_output, nodes[0] # list, tree

'''Huffman Decoder'''
def Huffman_decoding(encoded_data, huffman_tree):
    
    tree_head = huffman_tree
    decoded_output = []
    
    for x in encoded_data:
        for x1 in x:
            if x1 == '1':   # go right
                huffman_tree = huffman_tree.right   
            elif x1 == '0': # go left
                huffman_tree = huffman_tree.left
            try:
                if huffman_tree.left.symbol == None and huffman_tree.right.symbol == None:
                    pass
            except AttributeError:  # reach at a leaf node
                decoded_output.append(huffman_tree.symbol)  # obtain the symbol
                huffman_tree = tree_head
        
    #string = ''.join([str(item) for item in decoded_output])
    #decoded_output = np.reshape(decoded_output, (rows,cols))
    
    return decoded_output  # 2d array

'''Subsampling'''
def subsampling(image):
    
    down_sampled = image[::2,::2]
    #print("new table:\n", down_sampled)
    
    return down_sampled
  
'''Low pass filter function ''' 
def low_pass_function(image):

    # # Gaussian Filter (Smoothing)
    # kernel = np.array([[1, 2, 1],
    #                    [2, 4, 2],
    #                    [1, 2, 1]]) / 16

    # low pass filter
    kernel = np.ones((3, 3)) / 9

    convolved_image = signal.convolve2d(image, kernel)

    truncated_image = truncate_v2(convolved_image, kernel)

    low_pass_filtered_image = truncated_image

    return low_pass_filtered_image

'''Low pass filter continue..'''
def truncate_v2(image, kernel):

    m, n = kernel.shape
    m = int((m-1) / 2)

    for i in range(0, m):
        line, row = image.shape
        image = np.delete(image, line-1, 0)
        image = np.delete(image, row-1, 1)
        image = np.delete(image, 0, 0)
        image = np.delete(image, 0, 1)
    return image

'''Continue of low pass filter function..'''
def three_channel(image, func):
    return np.stack([func(image[:, :, 0]),  # Channel Blue
                     func(image[:, :, 1]),  # Channel Green
                     func(image[:, :, 2])], axis=2)  # Channel Red

  
'''Hierarchical Search'''  
def hierarchical_search(referenceFrame, targetFrame, block_size, k):
    
    # target's frame dimensions
    height, width = targetFrame.shape
    
    print("\nheight ",height)
    print("width ", width)
    
    #rows and cols
    print("Rows: ", len(referenceFrame))
    print("\nCols: ", len(referenceFrame[0]))
    
    #macroblocks_topl_corner_list = []  # macroblock's top left corner list
    best_mvs = []              # best motion vectors list
    new_macroblocks_list = []  # new macroblocks list according to motion vectors
    
    
    # We work 4 each macroblock in the target frame:
    #for y in range(0, height - 64, block_size):     # step = block_size
    y = 0
    for x in range(0, width - 64, block_size):  # step = block_size    # 2112 = 2160 - 64
            
            #top_left_corner = (x,y)
            #macroblocks_topl_corner_list.append(top_left_corner)  # append to list
            
            #get_searchArea(x, y, referenceFrame, block_size, k)
            
            r_frame_list = []  # reference frame in every level
            t_frame_list= []   # target frame in every level
            
            rframe_searchArea_list = []
            
            r_frame_list.append(referenceFrame) # initial reference frame
            t_frame_list.append(targetFrame)    # initial target frame
            
            level = 1  # start up level
            #while (level < 2):  # leveling up
            
            rframe_area = get_search_area(x, y, referenceFrame, block_size, k)[0]  # new reference frame area 
            rframe_searchArea_list.append(rframe_area) 
            #print("Rf area:\n", referenceFrame_area)
    
            minX = get_search_area(x, y, referenceFrame, block_size, k)[1]
            maxX = get_search_area(x, y, referenceFrame, block_size, k)[2]
        
            minY = get_search_area(x, y, referenceFrame, block_size, k)[3]
            maxY = get_search_area(x, y, referenceFrame, block_size, k)[4]
            
            # --- Deal with the motion vector in the highest level:
            print("\nMacroblock in target frame top left corner (x,y): ",(x,y))
            print("Block's size: ", block_size)
            print("About search area:")
            print("minX: ", minX)
            print("maxX: ", maxX)
            print("minY: ", minY)
            print("maxY: ", maxY)
            
            # Sumsampling  
            #referenceFrame = subsampling(referenceFrame)  # reference's frame dimensions redused by half
            #targetFrame = subsampling(targetFrame)        # target's frame dimensions redused by half
        
            #print("\nAfter sampling:")
            #print("Rf is (after sampl.):\n", referenceFrame)
            
            # Converting the numpy arrays into images
            #rf_image = Image.fromarray(np.uint8(referenceFrame))
            #tf_image = Image.fromarray(np.uint8(targetFrame))
            
            # Saving the images
            #rf_image.save("rf_image.png")
            #tf_image.save("tf_image.png")
            
            # Images into 2d arrays
            #rf_arr = imageio.v2.imread("rf_image.png")
            #tf_arr = imageio.v2.imread("tf_image.png")
            
            #print("\nRf after img convert.\n", rf_arr)
            #print("\nTf after img convert.\n", tf_arr)
        
            #print(referenceFrame==rf_arr)
        
            '''
            # Filter 1: Apply Gaussian blur filter (it's a low pass filter)
            ksize = 3
            sigmaX =1
            referenceFrame = cv2.GaussianBlur(rf_arr, (ksize, ksize), sigmaX) # type: ignore
            targetFrame = cv2.GaussianBlur(tf_arr, (ksize, ksize), sigmaX)    # type: ignore
        
            print("\nAfter low pass filter:")
            print("Rf:\n", referenceFrame)
            print("Tf:\n", targetFrame)
        
            # Converting the numpy arrays into images
            rf_image = Image.fromarray(referenceFrame)
            tf_image = Image.fromarray(targetFrame)
        
            # Saving the images
            rf_image.save("rf1_image.png")
            tf_image.save("tf1_image.png")
            '''
        
            # append new level frames into lists
            #r_frame_list.append(referenceFrame)
            #t_frame_list.append(targetFrame)
            
            #block_size = block_size // 2   # macroblock's size redused by half
            #k = k // 2                     # search area size redused by half
            #print("\nSearch area size (k): ", k)
            
            
            #size = rows * cols
            #print("Rows: ", rows)
            #print("Columns: ", cols)
            #print("Image's size: ", size)
            
            #level+=1
    
            #print("Number of levels: ", level)
            
            
            best_mv = get_motion_vector(referenceFrame, targetFrame, (x,y), block_size, k, minX, maxX, minY, maxY)
            best_mvs.append(best_mv)
            #print("\nBest motion vector (mv): ", best_mv)
    
            # find the new macroblock's top left corner with the help of the motion vector
            new_macroblock = motion_compensate(referenceFrame, (x, y), best_mv, block_size)
            
            new_macroblocks_list.append(new_macroblock)
            
            newX = new_macroblock[0]
            newY = new_macroblock[1] 
            
            pframe = pFrame(targetFrame, referenceFrame, x, y, newX, newY, block_size)
            #print("\nNew macroclock:\n", new_macroblock)
    
            #encoding_mv = encoder(best_mv)[0]
            #print("Encoding mv is:\n", encoding_mv)
    
            #decoding_mv = Huffman_decoding(encoding_mv, encoder(best_mv)[1])
            #print("\nDecoding mv is:\n", decoding_mv)
    
    print("\nEnd of hierarchical search.")
    
    print("\nENCODER....")
    errorImage = error_image(pframe, targetFrame)
    encoder(errorImage, best_mvs)
    
    return best_mvs, new_macroblocks_list, pframe
 
'''Get search area function''' 
def get_search_area(x, y, referenceFrame, block_size, k):
    '''
    Returns image of reference Frame search area
    
    -param x, y: top left coordinate of macroblock in target Frame
    -param referenceFrame: reference Frame
    -param blockSize: size of block in pixels
    -param k: size of search area in pixels
    
    -return: Image of reference Frame search area
    '''
    
    #print("Search area function here!")
    h, w = referenceFrame.shape 
    
    min_X = x - k
    if (min_X < 0):
        min_X = x #=0
        
    max_X = x + block_size + k
    if max_X > len(referenceFrame[0]):
        max_X = len(referenceFrame[0])
        
    min_Y = y - k 
    if (min_Y < 0):
        min_Y = y #=0
        
    max_Y = y + block_size + k
    if max_Y > len(referenceFrame):
        max_Y = len(referenceFrame)
    
    
    # slice reference frame within bounds to produce reference search area
    referenceSearch = referenceFrame[min_X:max_X, min_Y:max_Y]
        
    return referenceSearch, min_X, max_X, min_Y, max_Y


''' STELLA'S '''
def get_new_block_tlc(): # tlc : top left corner 
    
    return 0
    
'''Find the best motion vector''' 
def get_motion_vector(referenceFrame, targetFrame, macroblock, block_size, k, minX, maxX, minY, maxY): # k is the number of levels -> (2k + 1)^2 candidate motion vector positions -> SAD metrix
    
    x,y = macroblock
    
    best_mv = (0, 0)
    best_sad = float('inf')
   
    for j in range(-k, +k): 
        for i in range(-k, +k):
            
            # calculate SAD (Sum of Absolute Differences)
            sad = get_sad(referenceFrame, targetFrame, i, j, (x,y), block_size)
            
            # check if the current motion vector is the best so far
            if sad < best_sad:
                best_sad = sad
                best_mv = (i, j)
                
    print("best sad ", best_sad)
    print("best mv: ",best_mv)
   
    return best_mv
    
'''SAD Metric'''
def get_sad(referenceFrame, targetFrame, i, j, current_p, block_size):
    
    current_x, current_y = current_p
    sad = 0
    
    rows = len(referenceFrame)
    cols = len(referenceFrame[0])
        
    #print("Rows: ", rows)
    #print("Columns: ", cols)
   
    for p in range(current_y, current_y + block_size):
        for q in range(current_x, current_x + block_size):
            sad += (np.abs(targetFrame[p,q] - referenceFrame[p+i, q+j]))
    
    #print(sad)
    return sad

'''Find the new macroblock according to motion vector'''
def motion_compensate(referenceFrame, position, motion_vector, block_size):
    # macroblock's position in current frame
    x, y = position
    #print("\nold x: ", x)
    #print("\nold y: ", y)
    
    
    newX = x + motion_vector[0]
    newY = y + motion_vector[1]
    #print("Ney x: ", newX)
    #print("New y: ", newY)
    
    
    # find the new macroblock
    #new_macroblock = referenceFrame[newX:newX + block_size, newY: newY+block_size]
    #referenceFrame[y:y+block_height, x:x+block_width] = referenceFrame[y:y+block_height, x:x+block_width] + motion_vector
    
    
    return (newX,newY)

'''Find the predicted frame'''
def pFrame(targetFrame, referenceFrame, x, y, newX, newY, block_size):
    
    targetFrame[x:x+block_size, y:y+block_size] = referenceFrame[newX:newX+block_size, newY:newY+block_size]
    
    return targetFrame

'''Find the error image'''
def error_image(pFrame, targetFrame):
    return pFrame - targetFrame

'''Encoder Function'''    
def encoder(error_image, mv_list):
    '''
    Has to:
    1. Compress motion vectors using Huffman encoding
    2. compress error image by using JPEG encoding
    '''
     # Encode the motion vector using Huffman encoding
    encoding_mv, tree = Huffman_encoding(mv_list) #convert 2d array to 1d
    print("Encoding mvs:\n", encoding_mv)
    
    # 6. Find the initial error image using Huffman decoding
    #decoding = Huffman_decoding(encoding,tree)
    #errorImage = tFrame - rFrame # encode this image with JPEG encoding
    #motionVectors = []           # encode using Huffman encoding
    
    return encoding_mv, tree


'''Decoder Function'''
def decoder(encodingErrorImage, encoding_mvs, tree, referenceFrame):
    '''
    Has to:
    1. Reconstruct Frame1 (n)   =  Reference Frame
    2. Reconstruct Frame2 (n+1) = Target Frame by:
       - Encoding motion vectors (Huffman encoding)
       - Encoding error image  given by the encoder function!
    '''
    
    decoding_mvs = Huffman_decoding(encoding_mvs, tree)
    print("\nDecoding mvs is:\n", decoding_mvs)
    
    #2d array with 0
    
    
    
    return 0

'''-------------------------------------------------------MAIN-------------------------------------------------------------------------------------------------'''
# 1. Collect video frames
#num_of_frames = collect_frames("video1.mp4")


image = get_image("./output/frame_0.jpg")  # reference frame
image = geek.array(image)

# find the rows and cols
rows = len(image)
cols = len(image[0])

print("\nFrame's dimensions")
print("--------------------")
print("Rows: ", rows)
print("Columns: ", cols)

print("\nReference frame is:\n", image)


image1 = get_image("./output/frame_1.jpg")   # target frame
image1 = geek.array(image1)
print("\nTarget frame is:\n", image1)
   


# Motion compensate with hierarchical search starts here:
block_size = 64  # macroblock's size
k = 32           # search range

best_mvs, new_mblock_list, pframe = hierarchical_search(image, image1, block_size, k)

#print("Macroblocks dim are:\n", l)

#print("\nNumber of motion vectors: ", len(best_mvs))
print("\nBest mvs list:\n", best_mvs)

print("\nNew macroblocks list:\n", new_mblock_list)

print("predicted frame is:\n", pframe)

# Converting the numpy array into image
#cv2.imwrite('SearchArea.png', s_image)  # reference's frame search area

# Load the image
#s_image = cv2.imread('SearchArea.png')

x = 128
y = 128
referenceFrame = image
get_search_area(x, y,image, block_size, k)

minX = get_search_area(x, y, referenceFrame, block_size, k)[1]
maxX = get_search_area(x, y, referenceFrame, block_size, k)[2]
                    
minY = get_search_area(x, y, referenceFrame, block_size, k)[3]
maxY = get_search_area(x, y, referenceFrame, block_size, k)[4]
            
'''          
print("\nMacroblock in target frame top left corner (x,y): ",(x,y))
print("Block's size: ", block_size)
print("About search area:")
print("minX: ", minX)
print("maxX: ", maxX)
print("minY: ", minY)
print("maxY: ", maxY)


mv = get_motion_vector(referenceFrame, image1, (x,y), block_size, k, minX, maxX, minY, maxY)
print("mv is: ", mv)
'''