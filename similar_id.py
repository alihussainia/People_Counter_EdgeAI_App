def is_same_person(p1, p2, max_percent):
    """
    Check if 2 persons (represented by their bounding box coordinates) can be considered as the same person.
    :param p1: first person
    :param p2: second person
    :return: True if they're the same person
    """
    #print('Comparing ', p1, " and ", p2)
    is_same = False
    # if there's not too much time this person have been detected
    if p1[5] - p2[5] < MAX_NB_FRAMES:
        # compute the coordinates of the centers of the bounding boxes
        center_x1 = (p1[0] + p1[2]) / 2
        center_y1 = (p1[1] + p1[3]) / 2
        center_x2 = (p2[0] + p2[2]) / 2
        center_y2 = (p2[1] + p2[3]) / 2
        # compute the distance between the two centers
        distance = sqrt((center_x1 - center_x2)**2 + (center_y1 - center_y2)**2)
        #print('distance=',distance)
        # for comparison, compute the distance of the diagonal of one bounding box
        diag = sqrt((p1[2] - p1[0])**2 + (p1[3] - p1[1])**2)
        # return true if the move is not greater than MAX_MOVE_PERCENTAGE percents of the bounding box diagonal
        #print('(distance / diag)=',(distance / diag))
        if (distance / diag) > max_percent[0] and (distance / diag) < MAX_MOVE_PERCENTAGE:
            max_percent[0] = (distance / diag)
        is_same = (distance / diag) < MAX_MOVE_PERCENTAGE
    return is_same