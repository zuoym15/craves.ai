import os, json
import matplotlib.pyplot as plt
import imageio

folder = 'C:/temp'
annotation_json = os.path.join(folder, 'object_list.json')
seg_img = os.path.join(folder, 'obj_mask.png')

with open(annotation_json, 'r') as f:
	obj_list = json.load(f)

print('Number of objects: %d' % len(obj_list))

# mask = plt.imread(seg_img)
mask = imageio.imread(seg_img)
print(mask.max())

obj_area = {}

# Match object with colors
for obj_name, annotation_color in obj_list.iteritems():
	c = annotation_color
	r, g, b = c["R"], c["G"], c["B"]

	obj_mask = (mask[:,:,0]==r)&(mask[:,:,1]==g)&(mask[:,:,2]==b)
	obj_area[obj_name] = obj_mask.sum()

	if obj_mask.sum() > 0:
		print(obj_name)
		print(r, g, b)
		print(obj_mask.sum())


seg_folder = 'C:/temp/seg'

print obj_area
total_area = sum([v for (k,v) in obj_area.iteritems()])
print(total_area)
print(mask.shape)
