class SubdivisionPathIterator():
    """
    An iterator that produces the subdivision selection of elements.
    
    a b c d e f g h i j k -> f c i a d g j b e h k
    """

    def __init__(self, local_path):
        self.segment_queue = [local_path]

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.segment_queue) == 0:
            raise StopIteration
        else:
            segment = self.segment_queue.pop(0)
            m_idx = int(len(segment) / 2)
            s1 = segment[:m_idx]
            s2 = segment[m_idx + 1:]
            if len(s1) > 0:
                self.segment_queue.append(s1)
            if len(s2) > 0:
                self.segment_queue.append(s2)
            if len(segment) > 1:
                return segment[m_idx]
            elif len(segment) == 1:
                return segment[0]

    next = __next__  # python2.x compatibility.


def incremental_evaluate(eval_fn, local_path):
    """
    Incrementally evaluates a discrete local path i.e. 1-2-3-4-5-6-7-8

    Args:
        eval_fn ([type]): Evaluating function.
        local_path ([type]): [description]

    Returns:
        [type]: [description]
    """
    for point in local_path:
        if not eval_fn(point):
            return False
    return True


def subdivision_evaluate(eval_fn, local_path):
    """
    Evaluates subdivisions of a descrete local path i.e. 4-2-5-1-6-3-7-8

    Args:
        eval_fn ([type]): [description]
        local_path ([type]): [description]

    Returns:
        [type]: [description]
    """
    for point in SubdivisionPathIterator(local_path):
        if not eval_fn(point):
            return False
    return True
