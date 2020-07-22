def publish_last_duration(previously_detected_persons, client, fps):
    '''
    Publish the duration for the last detected person
    '''
    # previous duration is last person "out" minus "in" frame, divided by fps rate
    if len(previously_detected_persons) > 0:
        previous_duration = (previously_detected_persons[-1][5] - previously_detected_persons[-1][4]) / fps
        client.publish("person/duration", json.JSONEncoder().encode({"duration": previous_duration}))