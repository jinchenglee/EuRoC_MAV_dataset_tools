import numpy as np

class point_id:
    """
    Always incrementing point id, to identify points used in map and frames.

    TODO: make this a singleton class.
    """
    def __init__(self):
        self.id = 0

    def incr(self): # Increment currnet ID by 1
        self.id += 1
        return self.id

    def get(self): # Return current ID
        return self.id

class points_2d:
    """
    Image points class.

    Dictionary for newly found points via optical flow in add(). Unique point ID
    should be provided when calling add(). Point ID is maintained by class point_id.
    """

    def __init__(self):
        self.pts = {} # Empty point dictionary

    def add(self, x, y, cur_id):
        if [x,y] in self.pts.values():
            print("Err: item [", x, ",", y, "] already exists in dictionary.")
        elif cur_id in self.pts.keys():
            print("Err: point id ", cur_id, " already exists in dictionary.")
        else:
            self.pts[cur_id] = [x,y]

    def rm_pt_id(self, pt_id): # Remove by point id
        if pt_id in self.pts.keys():
            # remove item from dictionary
            del self.pts[pt_id]
        else:
            print("Err: Cannot delete. No such item found in dict with point id: ", pt_id)

    def rm_xy(self, x, y): # Remove by (x,y) coordinates
        pt_id_key = ''
        for key in self.pts.keys():
            if self.pts[key]==[x,y]:
                pt_id_key = key
                break
        if pt_id_key=='':
            print("Err: Cannot delete. No such item found in dict:[", x, ",", y, "]")
        else:
            # remove item from dictionary
            del self.pts[pt_id_key]

    def list(self): # List all items in the dict print("id, [x,y] = ")
        for k, v in self.pts.items():
            print(k, "[",v[0],",",v[1],"]")

    def get_pt_id(self, x, y): # Return current pt_id to update global one
        pt_id_key = ''
        for key in self.pts.keys():
            if self.pts[key]==[x,y]:
                pt_id_key = key
                break
        if pt_id_key=='':
            print("Err: Cannot find such item found in dict:[", x, ",", y, "]")
        else:
            return pt_id_key

    def length(self): # Return pts dictionary len
        return len(self.pts)



