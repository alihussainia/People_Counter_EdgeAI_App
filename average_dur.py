def get_avg_duration(persons, fps):
    """
    Compute the average duration of detection for an array of persons
    :param persons: array of the detected persons
    :param fps: frame-per-second rate for the video
    :return: None
    """
    if len(persons) > 0:
        total_nb_frames = 0
        for person in persons:
            total_nb_frames = total_nb_frames + person[5] - person[4]   
        # return the average number of frames by person, divided by the FPS rate to get a value in seconds    
        return (total_nb_frames / len(persons)) / fps    
    else:
        return 0