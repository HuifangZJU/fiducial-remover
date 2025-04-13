from backgroundremover.bg import remove
import matplotlib.pyplot as plt
from PIL import Image
def remove_bg(src_img_path, out_img_path):
    model_choices = ["u2net", "u2net_human_seg", "u2netp"]
    f = open(src_img_path, "rb")
    data = f.read()
    img = remove(data, model_name=model_choices[0],
                 alpha_matting=True,
                 alpha_matting_foreground_threshold=240,
                 alpha_matting_background_threshold=10,
                 alpha_matting_erode_structure_size=10,
                 alpha_matting_base_size=1000)
    f.close()
    img[0].save(out_img_path)



inpath = "/media/huifang/data/fiducial/tiff_data/V1_Mouse_Brain_Sagittal_Posterior_Section_2_spatial/spatial/tissue_hires_image.png"
outpath = "/media/huifang/data/fiducial/tiff_data/V1_Mouse_Brain_Sagittal_Posterior_Section_2_spatial/spatial/test.png"
remove_bg(inpath,outpath)