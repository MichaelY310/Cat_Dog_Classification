import os
from PIL import Image

# 目标：让图片的长和宽都大于250，同时保持原来的比例

print("resizing pictures...")
datafolder = os.path.join(".//train")
pictures = os.listdir(datafolder)
for picture in pictures:
    picture_dir = datafolder + "//" + picture
    print(picture_dir)
    # print(picture_dir)
    image = Image.open(picture_dir)
    if image.size[0] > image.size[1]:
        image = image.resize((250 * image.size[0] // image.size[1], 250))
    else:
        image = image.resize((250, 250 * image.size[1] // image.size[0]))
    image.save(picture_dir)

print('resize complete')