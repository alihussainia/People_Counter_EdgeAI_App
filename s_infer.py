#import
from argparse import ArgumentParser
from output_img import create_output_image
from average_dur import get_avg_duration
from p_img import preprocess_image
from last_dur import publish_last_duration


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.
    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # check if we provided a TF model or an IR
    is_tensorflow = os.path.splitext(args.model)[1] == '.pb'
    
    # Initialise the class
    if is_tensorflow:
        from inference_tf import NetworkTf
        infer_network = NetworkTf()
    else:
        from inference import Network
        infer_network = Network()
        
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### Load the model through `infer_network` ###
    infer_network.load_model(args.model, args.device, args.cpu_extension)
    
    if not is_tensorflow:
        net_input_shape = infer_network.get_input_shape()

    ### Handle the input ###
    is_single_image_mode = os.path.splitext(args.input)[1] in ['.jpg','.png']

    if not is_single_image_mode:
        cap = cv2.VideoCapture(args.input)
        cap.open(args.input)
        # Grab the shape and FPS rate of the input 
        width = int(cap.get(3))
        height = int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS)

    # init the total of detected persons
    total = 0
    
    # init the number of frames
    nb_frames = 0
    
    # init the total inference time, to be divided by the number of frames at the end
    total_inference_time = 0
    
    # an array to keep track of previously detected persons
    previously_detected_persons = []
    
    # the average duration of a single person presence
    duration = 0
    
    # fixme  : for debugging
    max_percent = [0]
    
    ### Loop until stream is over ###
    while is_single_image_mode or cap.isOpened():

        if is_single_image_mode:
            print("Single image mode. Analyze ", args.input)
            frame = cv2.imread(args.input)
            height = frame.shape[0]
            width = frame.shape[1]
        else:
            ### Read from the video capture ###
            flag, frame = cap.read()
            nb_frames = nb_frames + 1
            if not flag:
                break
            key_pressed = cv2.waitKey(60)

        if is_tensorflow:
            p_frame = frame
        else:    
            p_frame = preprocess_image(frame, (net_input_shape[3], net_input_shape[2]))
        
        ### Start asynchronous inference for specified request ###
        inference_start = time.time()
        infer_network.exec_net(p_frame)

        ### Wait for the result ###
        if infer_network.wait() == 0:
            
            # record the inference time
            total_inference_time = total_inference_time + (time.time() - inference_start)

            ### Get the results of the inference request ###
            result = infer_network.get_output()
            
            ### Create output frame
            out_frame, detected_persons = create_output_image(frame, result, width, height, (0, 0, 255), float(args.prob_threshold), nb_frames)
            
            if not is_single_image_mode:
                # if there's detected persons in the frame
                count = 0
                if len(detected_persons) > 0:
                    # for each new detection
                    for person in detected_persons:
                        # check if there was a person with a matching bounding box
                        is_new_person = True
                        for index, previous_person in enumerate(previously_detected_persons):
                            # if same person, updating to last coords
                            if is_same_person(person, previous_person, max_percent):
                                # keep the timestamp of the first detection
                                person[4] = previous_person[4]
                                previously_detected_persons[index] = person
                                is_new_person = False
                                break
                        if is_new_person:
                            total = total + 1
                            publish_last_duration(previously_detected_persons, client, fps)
                            previously_detected_persons.append(person)

                #print('previously_detected_persons=',previously_detected_persons)
                #print('max_percent=',max_percent)

                ### Extract any desired stats from the results ###

                ### Calculate and send relevant information on ###
                ### current_count, total_count and duration to the MQTT server ###
                ### Topic "person": keys of "count" and "total" ###
                ### Topic "person/duration": key of "duration" ###
                duration = get_avg_duration(previously_detected_persons, fps)
                #print('count:', len(detected_persons), " total:", len(previously_detected_persons), " person/duration:", duration)
                client.publish("person", json.JSONEncoder().encode({"count": len(detected_persons), "total": len(previously_detected_persons)}))
            
        ### Write an output image if is in single_image_mode ###
        if is_single_image_mode:
            print("Write output file in 'single_image.png'")
            cv2.imwrite('single_image.png', out_frame)
        else:
            ### Send the frame to the FFMPEG server ###
            sys.stdout.buffer.write(out_frame)
            sys.stdout.flush()

        # Break if single_image_mode or escape key pressed
        if is_single_image_mode or key_pressed == 27:
            break
            
    # publish duration for the last detected person
    if not is_single_image_mode:
        publish_last_duration(previously_detected_persons, client, fps)
    
    # Release the capture and destroy any OpenCV windows
    if not is_single_image_mode:
        cap.release()
        cv2.destroyAllWindows()
    
    # Debug mode : Print end results
    #print("Total count of persons : ", total)
    #print("Average presence time : ", duration)
    #print("Number of frames : ", nb_frames)
    #print("Total inference time : ", total_inference_time)
    #print("Inference time by frame : ", total_inference_time / nb_frames)










