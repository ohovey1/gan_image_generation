import numpy as np

def load_model():
    '''
    This function is used to load model, codes below are based on template.py.
    Please modify this function based on your own codes.
    '''
    #TODO: Implement
    pass

def generate_image(model):
    '''
    Take the model as input and generate one image, codes below are based on template.py.
    Please modify this function based on your own codes.
    '''
    # Set the dimensions of the noise
    z_dim = 100
    z = np.random.normal(size=[1, z_dim])
    generated_images = model.predict(z)
    return generated_images

if __name__ == "__main__":
    model = load_model()
    image = generate_image(model)