def indexify(cls):
    """
    Function for modifying the Dataset class to return a tuple data, target, index instead of just data, target.
    """

    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        return data, target, index

    return type(cls.__name__, (cls,), {'__getitem__': __getitem__,})
