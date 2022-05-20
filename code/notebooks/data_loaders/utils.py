class Utilities():
    
    # Function to normalize pixels
    @staticmethod 
    def norm_pixel(x):
        x = x.astype('float32')
        return (x*2)/255-1

    # Function to denormalize pixels
    @staticmethod 
    def denorm_pixel(x):
        x = x.astype('float32')
        return (x+1)*255/2

    # Function to normalize pitch and roll angles
    @staticmethod
    def norm_pr(x, min = -90.0, max = 90.0):
        return ((x - min) * 2) / (max - min) - 1

    # Function to denormalize pitch and roll angles
    @staticmethod
    def denorm_pr(x, min = -90.0, max = 90.0):
        return ((x + 1) * (max - min))/2 + min
    
