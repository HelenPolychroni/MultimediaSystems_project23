import cv2
import numpy
import numpy as geek
import numpy as np
from scipy import ndimage, signal
from PIL import Image
import matplotlib.pyplot as mp
import imageio
from scipy.fftpack import dct
import matplotlib.pyplot as plt

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
    for couple in data:
        for element in couple:
            if symbols.get(element) == None:
                symbols[element] = 1
            else: 
                symbols[element] += 1 
                
    return symbols

    
'''Function to obtain the encoded output''' 
def output_encoded(data, coding):
   
    encoding_output = []
    for d in data:
        for a in d:
            encoding_output.append(coding[a])
    
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
        
    decoded_output_ = zip(decoded_output[0::2], decoded_output[1::2])
    decoded_output__ = []
    for pairs in decoded_output_:
        decoded_output__.append(pairs)
        
    return decoded_output__ 

def downsampling(image):
    
    down_sampled=image[::2,::2]
    down_image = Image.fromarray(np.uint8(down_sampled))
    down_image.save("down_image3333.png")
    
    print("NEW Rows: ", len(down_sampled))
    print("NEW Cols: ", len(down_sampled[0]))
    print("DOWNSAMPLED\n", down_sampled)
    
    '''
    down_sampled = image[::2,::2]
    print("new table:\n", down_sampled)
    print("NEW Rows: ", len(down_sampled))
    print("NEWCols: ", len(down_sampled[0]))
    '''
    return down_sampled
 
def upsampling(image):
    
    up_sampled=ndimage.zoom(image,2,order=0)
    up_image = Image.fromarray(np.uint8(up_sampled))
    
    print("NEW Rows upp: ", len(up_sampled))
    print("NEW Cols: upp", len(up_sampled[0]))
    print("UPSAMPLED\n", up_sampled)
    
    return up_sampled

'''Hierarchical Search'''  
def hierarchical_search(referenceFrame, targetFrame, block_size, k):
    
    # target's frame dimensions
    height, width = targetFrame.shape
    
    print("\nHeight ",height)
    print("Width ", width)
    print("-------------")
    #rows and cols
    print("Rows: ", len(referenceFrame))
    print("Cols: ", len(referenceFrame[0]))
        
    # We work 4 each macroblock in the target frame:
    best_mvs=[] # best motion vectors list
    for y in range(0,height-block_size, block_size):
      for x in range(0,width-block_size,block_size):
        l=1
        k=32
        while (l<4):
         print("Level is:",l)
         referenceFrame=downsampling(referenceFrame)
         targetFrame = downsampling(targetFrame) 
         
         x=x//2
         y=y//2
         
         #Filter 1: Apply Gaussian blur filter (it's a low pass filter)
         ksize = 3
         sigmaX =1
         referenceFrame = cv2.GaussianBlur(rf_arr, (ksize, ksize), sigmaX) # type: ignore
         targetFrame = cv2.GaussianBlur(tf_arr, (ksize, ksize), sigmaX)    # type: ignore
        
         
         k=k//2
         block_size=block_size//2
         print("K is",k)
         #tf_image1.save("tf_image1.png")
         l+=1
        
        #data from top level
        start_row = get_search_area(x, y, referenceFrame, block_size, k)[1]
        finish_row = get_search_area(x, y, referenceFrame, block_size, k)[2]
        start_col = get_search_area(x, y, referenceFrame, block_size, k)[3]
        finish_col = get_search_area(x, y, referenceFrame, block_size, k)[4]
            
        # --- Deal with the motion vector in the highest level:
        print("\nMacroblock in target frame top left corner (x,y): ",(x,y))
        print("Block's size: ", block_size)
        print("\nAbout search area:")
        print("start row: ", start_row)
        print("finish row: ", finish_row)
        print("start column: ", start_col)
        print("finish column: ", finish_col)

        best_mv = get_motion_vector(referenceFrame, targetFrame, (x,y), block_size, k, start_row, finish_row, start_col, finish_col,(0,0))
     
        while(l>1):
        
         k=k*2
         block_size=block_size*2
         x=x*2
         y=y*2
         
         upsampling(referenceFrame)
         upsampling(targetFrame) #l-1 level
         
         best_mv=tuple(x * 2 for x in best_mv) #double the coordinates of the best motion vector
         #call motion compensate
         new_c=motion_compensate((x,y),best_mv)
         new_x=new_c[0]
         new_y=new_c[1]
         
         candidate_pos=[]
         
         new_c1=(new_x-1,new_y)
         candidate_pos.append(new_c1)
         
         new_c2=(new_x+1,new_y)
         candidate_pos.append(new_c2)
        
         new_c3=(new_x,new_y-1)
         candidate_pos.append(new_c3)
        
         new_c4=(new_x,new_y+1)
         candidate_pos.append(new_c4)
         
         new_c5=(new_x-1,new_y-1)
         candidate_pos.append(new_c5)
         
         new_c6=(new_x-1,new_y+1)
         candidate_pos.append(new_c6)
         
         new_c7=(new_x+1,new_y-1)
         candidate_pos.append(new_c7)
         
         new_c8=(new_x+1,new_y+1)
         candidate_pos.append(new_c8)
         
         for i in range(len(candidate_pos)):
          start_row = get_search_area(x, y, referenceFrame, block_size, k)[1]
          finish_row = get_search_area(x, y, referenceFrame, block_size, k)[2]
        
          start_col = get_search_area(x, y, referenceFrame, block_size, k)[3]
          finish_col = get_search_area(x, y, referenceFrame, block_size, k)[4]
          
          best_mv1 = get_motion_vector(referenceFrame, targetFrame, (x,y), block_size,k, start_row, finish_row, start_col, finish_col, best_mv)
          
         l-=1
         
        best_mvs.append(best_mv1)
        
        new_macroblock = motion_compensate((x, y), best_mv1)
        newX = new_macroblock[0]
        newY = new_macroblock[1] 
        
        pframe = pFrame(targetFrame, referenceFrame, x, y, newX, newY, block_size)
    
    #end of loop
    
    print("\nEnd of hierarchical search.")
    
    return best_mvs, pframe


 
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
    
    start_row = x - k
    if (start_row < 0):
        start_row = x #=0
        
    finish_row = x + block_size + k
    if finish_row >  len(referenceFrame):
        finish_row = len(referenceFrame)
        
    start_col = y - k 
    if (start_col < 0):
        start_col = y #=0
        
    finish_col = y + block_size + k
    if finish_col >  len(referenceFrame[0]):
        finish_col = len(referenceFrame[0])
    
    
    # slice reference frame within bounds to produce reference search area
    referenceSearch = referenceFrame[start_row:finish_row, start_col:finish_col]
    
    return referenceSearch, start_row, finish_row, start_col, finish_col

'''Find the best motion vector''' 
def get_motion_vector(referenceFrame, targetFrame, macroblock, block_size, k, start_row, finish_row, start_col, finish_col,c): # k is the number of levels -> (2k + 1)^2 candidate motion vector positions -> SAD metrix
    
    x,y = macroblock   #(0,0)
    
    best_mv = c
    best_sad = float('inf')
   
    for row in range(start_row, finish_row):         # in search area
        for column in range(start_col, finish_col):  # in search area
            
            # calculate SAD (Sum of Absolute Differences)
            sad = get_sad(referenceFrame, targetFrame, row, column, (x,y), block_size)
                
            # check if the current motion vector is the best so far
            if sad < best_sad:
                best_sad = sad
                best_mv = (row, column)
                
    print("\nBest sad ", best_sad)
    print("Best motion vector: ",best_mv)
   
    return best_mv
    
'''SAD Metric'''
def get_sad(referenceFrame, targetFrame, row, column, current_p, block_size):
    
    current_x, current_y = current_p
    sad = 0
    
    rows = len(referenceFrame)
    cols = len(referenceFrame[0])
        
    #print("Rows: ", rows)
    #print("Columns: ", cols)
   
    for p in range(current_y, current_y + block_size):
        for q in range(current_x, current_x + block_size):
            sad += (np.abs(targetFrame[p,q] - referenceFrame[row, column]))
    
    #print(sad)
    return sad

'''Find the new macroblock according to motion vector'''
def motion_compensate(position, motion_vector):
    
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


'''
jpeg encoder Function
'''
def jpeg_encoder(frame):
    #convert to YCbCr color space
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    
    # Convert the array to a numpy array
    frame1 = np.array(frame, dtype=np.uint8)

    # Create an image from the array
    image = Image.fromarray(frame1)

    # Save the image as a JPEG file
    image.save('frame.jpg')   
    
    frame = cv2.imread("frame.jpg", cv2.IMREAD_COLOR)

    #split frame to Y, Cr, Cb channels
    Y, Cr, Cb = cv2.split(frame)

    #quantization matrices
    quantizationMatrix_Y = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                  [12, 12, 14, 19, 26, 58, 60, 55],
                                  [14, 13, 16, 24, 40, 57, 69, 56],
                                  [14, 17, 22, 29, 51, 87, 80, 62],
                                  [18, 22, 37, 56, 68, 109, 103, 77],
                                  [24, 35, 55, 64, 81, 104, 113, 92],
                                  [49, 64, 78, 87, 103, 121, 120, 101],
                                  [72, 92, 95, 98, 112, 100, 103, 99]])

    quantizationMatrix_Cr_Cb = np.array([[17, 18, 24, 47, 17, 18, 24, 47], 
                                      [18, 21, 26, 66, 18, 21, 26, 66],
                                      [24, 26, 56, 99, 24, 26, 56, 99],
                                      [47, 66, 99, 99, 47, 66, 99, 99],
                                      [17, 18, 24, 47, 17, 18, 24, 47],
                                      [18, 21, 26, 66, 18, 21, 26, 66],
                                      [24, 26, 56, 99, 24, 26, 56, 99],
                                      [47, 66, 99, 99, 47, 66, 99, 99]])


    #block size
    blockSize = quantizationMatrix_Y.shape[0]

    #number of blocks in each dimension
    blocksHeight = frame.shape[0] // blockSize
    blocksWidth = frame.shape[1] // blockSize

    #initialize an array filled with 0
    encodedFrame = np.zeros_like(frame, dtype=np.float32)

    for i in range(blocksHeight):
        for j in range(blocksWidth):
            #get current block
            blockY = Y[i*blockSize:(i+1)*blockSize, j*blockSize:(j+1)*blockSize]
            blockCr = Cr[i*blockSize:(i+1)*blockSize, j*blockSize:(j+1)*blockSize]
            blockCb = Cb[i*blockSize:(i+1)*blockSize, j*blockSize:(j+1)*blockSize]

            #downsample Cr, Cb 
            blockCr = cv2.resize(blockCr, (blockSize, blockSize), interpolation=cv2.INTER_LINEAR)
            blockCb = cv2.resize(blockCb, (blockSize, blockSize), interpolation=cv2.INTER_LINEAR)


            #dct on Y, Cr, and Cb blocks
            blockY = cv2.dct(blockY.astype(np.float32))
            blockCr = cv2.dct(blockCr.astype(np.float32))
            blockCb = cv2.dct(blockCb.astype(np.float32))

            #quantization on Y, Cr, and Cb blocks
            blockY = np.round(blockY / quantizationMatrix_Y)
            blockCr = np.round(blockCr / quantizationMatrix_Cr_Cb)
            blockCb = np.round(blockCb / quantizationMatrix_Cr_Cb)

            '''
            #entropy encoding
            blockY = Huffman_ecoding(blockY)
            blockCr = Huffman_ecoding(blockCr)
            blockCb = Huffman_ecoding(blockCb)
            '''
            #store each channel in the frame
            encodedFrame[i*blockSize:(i+1)*blockSize, j*blockSize:(j+1)*blockSize, 0] = blockY
            encodedFrame[i*blockSize:(i+1)*blockSize, j*blockSize:(j+1)*blockSize, 1] = blockCr
            encodedFrame[i*blockSize:(i+1)*blockSize, j*blockSize:(j+1)*blockSize, 2] = blockCb

    return encodedFrame


'''
jpeg decoder Function
'''
def jpeg_decoder(encodedFrame):

    quantizationMatrix_Y = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                  [12, 12, 14, 19, 26, 58, 60, 55],
                                  [14, 13, 16, 24, 40, 57, 69, 56],
                                  [14, 17, 22, 29, 51, 87, 80, 62],
                                  [18, 22, 37, 56, 68, 109, 103, 77],
                                  [24, 35, 55, 64, 81, 104, 113, 92],
                                  [49, 64, 78, 87, 103, 121, 120, 101],
                                  [72, 92, 95, 98, 112, 100, 103, 99]])

    quantizationMatrix_Cr_Cb = np.array([[17, 18, 24, 47, 17, 18, 24, 47],
                                      [18, 21, 26, 66, 18, 21, 26, 66],
                                      [24, 26, 56, 99, 24, 26, 56, 99],
                                      [47, 66, 99, 99, 47, 66, 99, 99],
                                      [17, 18, 24, 47, 17, 18, 24, 47],
                                      [18, 21, 26, 66, 18, 21, 26, 66],
                                      [24, 26, 56, 99, 24, 26, 56, 99],
                                      [47, 66, 99, 99, 47, 66, 99, 99]])


    #block size
    blockSize = quantizationMatrix_Y.shape[0]

    #number of blocks in each dimension
    blocksHeight = encodedFrame.shape[0] // blockSize
    blocksWidth = encodedFrame.shape[1] // blockSize

    #initialize an array filled with 0
    decodedFrame = np.zeros_like(encodedFrame)

    for i in range(blocksHeight):
        for j in range(blocksWidth):

            blockY = encodedFrame[i*blockSize:(i+1)*blockSize, j*blockSize:(j+1)*blockSize, 0]
            blockCr = encodedFrame[i*blockSize:(i+1)*blockSize, j*blockSize:(j+1)*blockSize, 1]
            blockCb = encodedFrame[i*blockSize:(i+1)*blockSize, j*blockSize:(j+1)*blockSize, 2]
            
            '''
            #entropy decoding
            blockY = Huffman_decoding(blockY)
            blockCr = Huffman_decoding(blockCr)
            blockCb = Huffman_decoding(blockCb)
            '''

            #upsample Cr, Cb
            blockCr = cv2.resize(blockCr, (blockSize, blockSize), interpolation=cv2.INTER_LINEAR)
            blockCb = cv2.resize(blockCb, (blockSize, blockSize), interpolation=cv2.INTER_LINEAR)

            #inverse quantization
            blockY = blockY * quantizationMatrix_Y
            blockCr = blockCr * quantizationMatrix_Cr_Cb
            blockCb = blockCb * quantizationMatrix_Cr_Cb

            #inverse DCT
            blockY = cv2.idct(blockY)
            blockCr = cv2.idct(blockCr)
            blockCb = cv2.idct(blockCb)

            #store each channel in the frame
            decodedFrame[i * blockSize:(i + 1) * blockSize, j * blockSize:(j + 1) * blockSize, 0] = blockY
            decodedFrame[i * blockSize:(i + 1) * blockSize, j * blockSize:(j + 1) * blockSize, 1] = blockCr
            decodedFrame[i * blockSize:(i + 1) * blockSize, j * blockSize:(j + 1) * blockSize, 2] = blockCb


    #convert the frame to RGB color space
    #decodedFrame = cv2.cvtColor(decodedFrame, cv2.COLOR_YCR_CB2BGR)

    return decodedFrame

'''Encoder Function'''    
def encoder(error_image, mv_list):
    '''
    Has to:
    1. Compress motion vectors using Huffman encoding
    2. compress error image by using JPEG encoding
    '''
    
    # Encode the motion vector using Huffman encoding
    encoding_mv, tree = Huffman_encoding(mv_list) 
    print("Encoding mvs:\n", encoding_mv)
    
    # Encode the error image using jpeg encoding
    encoding_image = jpeg_encoder(error_image)
    
    return encoding_mv, tree, encoding_image


'''Decoder Function'''
def decoder(encodingErrorImage, encoding_mvs, tree, referenceFrame):
    '''
    Has to:
    1. Reconstruct Frame1 (n)   =  Reference Frame
    2. Reconstruct Frame2 (n+1) = Target Frame by:
    '''
    #decode mvs
    decoding_mvs = Huffman_decoding(encoding_mvs, tree)
    print("\nDecoding mvs is:\n", decoding_mvs)
    #print("length mvs: ", len(decoding_mvs))
    
    # decode errorImage
    decodingErrorImage = jpeg_decoder(encodingErrorImage)
    
    r = np.zeros((rows, cols), dtype=int)  # reconstructed frame
    
    height, width = r.shape
    block_size = 64
    i = 0
    
    for y in range(0, height - 64, block_size):     # step = block_size
        for x in range(0, width - 64, block_size):  # step = block_size    
            new_x = decoding_mvs[i][0] 
            new_y = decoding_mvs[i][1] 
            r[y:y+block_size, x:x+block_size] = referenceFrame[new_x:new_x+block_size, new_y:new_y+block_size]
            i+=1
            
    r = r + decodingErrorImage
     
    print("New r (reconstructed frame n+1): ", r)   
    return r

'''-------------------------------------------------------MAIN-------------------------------------------------------------------------------------------------'''
# 1. Collect video frames
num_of_frames = collect_frames("video1.mp4")

image = get_image("./output/frame_0.jpg")  # reference frame
image = geek.array(image)

# find the rows and cols
rows = len(image)
cols = len(image[0])

block_size = 64  # macroblock's size
k = 32           # search range

for i in range(num_of_frames - 1):
    
    image1 = get_image(f"./output/frame_{i}.jpg")
    image2 = get_image(f"./output/frame_{i+1}.jpg")
    
    image1 = geek.array(image1)
    image2 = geek.array(image2)
    
    
    print("Reference Frame:\n",image1)
    print("\nTarget Frame:\n",image2)
    
    best_mvs, pframe = hierarchical_search(image1, image2, block_size, k)
    
    #Find the error image
    errorImage = error_image(pframe, image2)
    print("Error image table:\n", errorImage)
    
    print("ENCODER...")
    encoder(errorImage, best_mvs)   
    e_mvs = encoder(errorImage, best_mvs)[0]
    tree  = encoder(errorImage, best_mvs)[1]   
    encoding_image = encoder(errorImage, best_mvs)[2]   
    
    print("DECODER...")
    final_array = decoder(encoding_image, e_mvs, tree, image1)




