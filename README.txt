Behold my cat/dog classifier!!!!!!!!!!!!

Contents:
1. classify.py: uses an existing nn to classify provided images as cats or dogs
2. make_nn.py: uses a labeled image pool to make a nn that can classify between cats or dogs
3. pre_process_misc.py: copies and converts images en-masse to 100 x 100 pixels (DEPRICATED)
4. example.h5: an example network trained on a large number of cat and dog images

Required Packages:
1. tensorflow
2. numpy
3. os
4. sys
5. cv2

Using classify.py:
    Command will look like: 
                       python classify.py <neural_net> ls <img1> <img2> <img3> <img...>
                       OR
                       python classify.py <neural_net> dir <dir_w_images_to_clasify>
        
        The ls argument will allow you to list the filepaths to any number of individual images to classify
        The dir argument will allow you to put the filepath of a single directory and all images contained will be classified

        NOTE: all provided images will be copied and resized to be 100 x 100 pixels

Using make_nn.py:
    Command will look like:
                       python make_nn.py <img_sources> <network_name>

        All images in the img_sources directory must start with a c (case sensitive) to be labled a cat image 
        All images that do not begin with c (case sensitive) will be labeled a dog image
        All files that do not end with .jpg will be ignored

        NOTE: all provided images will be copied and resized to be 100 x 100 pixels

Using pre_process_misc.py:
    Command will look like:
                       python pre_process_misc.py <source_dir> <destination_dir>

        NOTE: all provided images will be copied and resized to be 100 x 100 pixels
    