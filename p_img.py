def preprocess_image(image, shape):
    '''
    Pre-process the image as needed
    ''' 
    p_frame = cv2.resize(image, shape)
    p_frame = p_frame.transpose((2,0,1))
    p_frame = p_frame.reshape(1, *p_frame.shape)
    return p_frame