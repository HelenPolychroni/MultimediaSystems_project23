import cv2
import numpy
import numpy as np
import numpy as geek
from PIL import Image

def get_image(image_path):
    '''
    Get a numpy array of an image so that one can access values[x][y].
    '''
    image = Image.open(image_path, "r")
    width, height = image.size
    pixel_values = list(image.getdata())
   
    if image.mode == "RGB":
        channels = 3
    elif image.mode == "L":  # greyscale image
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
  
'''Function to predict P Frame'''  
def predictPFrame(Aframe, BFrame):
    '''
    P (x,y) = A (x,y) + (B (x,y) - A (x,y))
    '''
    
    pFrame = Aframe + (BFrame - Aframe)
    
    return pFrame
    
      
# A Huffman Tree Node
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
    
'''
Encoder
'''
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
    #print("\nSymbols with codes:\n", huffman_encoding)
    
    encoded_output = output_encoded(data, huffman_encoding)
    #print("\nEncoded output:", encoded_output)
    
    return encoded_output, nodes[0] # list, tree

'''
Decoder
'''
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
    decoded_output = np.reshape(decoded_output, (rows,cols))
    
    return decoded_output  # 2d array
    
'''--------------------------------------MAIN----------------------------------'''    

# 1. Collect video frames
num_of_frames = collect_frames("video1.mp4")


image = get_image("./output/frame_0.jpg")
image = geek.array(image)
#print("frame0:\n",image)

# Find the rows and cols
rows = len(image)
cols = len(image[0])


sum = 0
for k in range(num_of_frames - 1):
    
    image1 = get_image(f"./output/frame_{k}.jpg")
    image2 = get_image(f"./output/frame_{k+1}.jpg")
    
    image1 = geek.array(image1)
    image2 = geek.array(image2)
    
    
    print("Reference Frame:\n",image1)
    print("\nTarget Frame:\n",image2)
    
    
    # 3. Find the predict p frame
    pFrame = predictPFrame(image1, image2)
    print("\nPredict PFrame is:\n", pFrame)
    
    
    # 4. Find the error image
    error = np.zeros((rows, cols))  # error matrix
    
    error = pFrame - image1   # type: ignore # Update error matrix with the difference between pframe and image1
    print("\nError table:\n",error) 
    
    # 5. Encode the error image using Huffman encoding
    encoding, tree = Huffman_encoding(error.flatten()) #convert 2d array to 1d
    
    # 6. Find the initial error image using Huffman decoding
    decoding = Huffman_decoding(encoding,tree)
    
    if (error.all() == decoding.all()): # original error matrix == final error matrix
        sum+=1
        
#print("End of loop.")  
#print("Sum is: ",sum)     
 
if sum == (num_of_frames - 1): # Decoder works fine for all error images
    print("Success!")  
else:
    print("Failure.")  
    
#print("Num of frames is: ",num_of_frames)

