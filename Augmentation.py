import Augmentor

p = Augmentor.Pipeline(r"dataset\Ansh")

#p.rotate90(probability=0.5)
#p.rotate270(probability=0.5)
#p.flip_left_right(probability=0.8)
#p.flip_top_bottom(probability=0.3)
#p.random_distortion(probability=1, grid_height=5,grid_width=5, magnitude=8)
#p.random_distortion(probability=1, grid_height=5,grid_width=5, magnitude=5)
#p.random_contrast(probability=1,min_factor=0.5,max_factor=0.5)
#p.black_and_white(probability=1)
#p.random_color(probability=1,min_factor=0.5,max_factor=0.5)
#p.crop_random(probability=1, percentage_area=0.5)
#p.invert(probability=1.0)
p.flip_random(probability=1)
p.resize(probability=1.0, width=256, height=256)

p.sample(20)