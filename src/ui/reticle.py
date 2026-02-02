import PIL.Image, PIL.ImageDraw, PIL.ImageFont
import numpy as np
from scipy.spatial.transform import Rotation
from data_access.frame_data import TrackerData

def tracker_to_4x4(x: TrackerData) -> np.ndarray:
    output = np.eye(4)
    output[0, 0] = x.r00
    output[0, 1] = x.r01
    output[0, 2] = x.r02
    output[1, 0] = x.r10
    output[1, 1] = x.r11
    output[1, 2] = x.r12
    output[2, 0] = x.r20
    output[2, 1] = x.r21
    output[2, 2] = x.r22
    output[0, 3] = x.tx
    output[1, 3] = x.ty
    output[2, 2] = x.tz

    return output

def relative_transform(target: TrackerData, actual: TrackerData) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute transformation parameters that transform actual to target.
    """
    target = tracker_to_4x4(target)
    actual = tracker_to_4x4(actual)
    
    # Translation is directly in the matrix (world space)
    translation = target[:3, 3] - actual[:3, 3]

    # Compute relative rotation: T = target @ inv(actual)
    R = target[:3, :3] @ np.linalg.inv(actual[:3, :3])
    
    # Convert to Euler angles: yaw(y), pitch(x), roll(z)
    rotation = Rotation.from_matrix(R)
    euler_angles = rotation.as_euler('XYZ', degrees=True)  # returns [yaw, pitch, roll]
    
    return translation, euler_angles

def apply_transform(pose: np.ndarray, translation: np.ndarray, euler: np.ndarray) -> np.ndarray:
    """
    Apply rotation then world-space translation to a pose.
    """
    T = np.eye(4)
    R = Rotation.from_euler('XYZ', euler, degrees=True).as_matrix()

    T[:3, :3] = R @ pose[:3, :3]
    T[:3, 3]  = pose[:3, 3] + translation
    
    # Apply: first rotate, then translate (in world space)
    return T


def clamp(x, low, high):
    if x < low:
        return low
    if x > high:
        return high
    return x

def render_pose(target, actual):
    translation, rotation = relative_transform(target, actual)

    image_txy = PIL.Image.new(mode='RGB', size=(448,448), color='white')
    image_tz  = PIL.Image.new(mode='RGB', size=(64,448), color='white')
    image_ryp = PIL.Image.new(mode='RGB', size=(256,256), color='white')
    image_rr  = PIL.Image.new(mode='RGB', size=(256,256), color='white')
    drawer_txy = PIL.ImageDraw.Draw(image_txy)
    drawer_tz  = PIL.ImageDraw.Draw(image_tz)
    drawer_ryp = PIL.ImageDraw.Draw(image_ryp)
    drawer_rr  = PIL.ImageDraw.Draw(image_rr)

    ## draw trans xy
    width  = image_txy.width
    height = image_txy.height
    drawer = drawer_txy
    xy_bbox_size = .2
    x = clamp((-translation[0]/xy_bbox_size*.5+.5)*width,  0, width-1)
    y = clamp(( translation[1]/xy_bbox_size*.5+.5)*height, 0, height-1)
    fill_color = 'red', 'lightcoral'
    if abs(x - width//2) < width//16 and abs(y - height//2) < height//16:
        fill_color = 'green', 'lightgreen'
    # draw correction
    drawer.line((x, y, width//2, height//2), fill=fill_color[1], width=6)
    # draw guidelines
    drawer.line((0, height//2, width, height//2), fill='lightgrey', width=1)
    drawer.line((width//2,  0, width//2, height), fill='lightgrey', width=1)
    drawer.circle((width//2, height//2), radius=4, fill='grey') # target dot
    # draw actual
    drawer.circle((x, y), radius=8, fill=fill_color[0])
    # draw bbox
    drawer.rectangle((0, 0, width-1, height-1), fill=None, outline='black', width=1)


    ## draw trans z
    width  = image_tz.width
    height = image_tz.height
    drawer = drawer_tz
    z_bbox_size = .1
    z = clamp((translation[2]/z_bbox_size*.5+.5)*height, 3, height-4)
    fill_color = 'red', 'lightcoral'
    if abs(z - height//2) < height//16:
        fill_color = 'green', 'lightgreen'
    # draw correction
    drawer.line((width//2, z, width//2, height//2), fill=fill_color[1], width=32)
    # draw guidelines
    drawer.line((0, height//2, width, height//2), fill='grey', width=3)
    # draw actual
    drawer.line((0, z, width, z), width=5, fill=fill_color[0])
    # draw bbox
    drawer.rectangle((0, 0, width-1, height-1), fill=None, outline='black', width=1)
    

    ## draw rr
    width  = image_rr.width
    height = image_rr.height
    drawer = drawer_rr
    r_bbox_size = 180
    roll = rotation[2]
    fill_color = 'red', 'lightcoral'
    if abs(roll) < 15:
        fill_color = 'green', 'lightgreen'
    # draw correction
    if roll > 0:
        drawer_rr.arc(((8, 8),(width-8, height-8)), start=-roll, end=0, fill=fill_color[1], width=5)
        drawer_rr.arc(((8, 8),(width-8, height-8)), start=-roll + 180, end=180, fill=fill_color[1], width=5)
    else:
        drawer_rr.arc(((8, 8),(width-8, height-8)), start=0, end=-roll, fill=fill_color[1], width=5)
        drawer_rr.arc(((8, 8),(width-8, height-8)), start=180, end=-roll + 180, fill=fill_color[1], width=5)
    # draw guidelines
    drawer.circle((width//2, height//2), radius=width//2 - 16, width=2, fill=None, outline='lightgrey')
    drawer.line((16, height//2, width-16, height//2), fill='lightgrey', width=3)
    # draw actual
    drawer.line((-np.cos(roll*3.14159/180)*(width//2-18) + width//2, np.sin(roll*3.14159/180)*(height//2 - 18) + height//2, np.cos(roll*3.14159/180)*(width//2-18) + width//2, -np.sin(roll*3.14159/180)*(height//2 - 18) + height//2), width=4, fill=fill_color[0])
    # draw bbox
    drawer.rectangle((0, 0, width-1, height-1), fill=None, outline='black', width=1)


    ## draw ryp
    width  = image_ryp.width
    height = image_ryp.height
    drawer = drawer_ryp
    yp_bbox_size = 30
    yaw   = clamp((-rotation[1]/yp_bbox_size*.5+.5)*height, 0, height-1)
    pitch = clamp(( rotation[0]/yp_bbox_size*.5+.5)*width,  0, width-1)
    fill_color = 'red', 'lightcoral'
    if abs(yaw - width//2) < width//16 and abs(pitch - height//2) < height//16:
        fill_color = 'green', 'lightgreen'
    # draw correction
    drawer.line((yaw, pitch, width//2, height//2), fill=fill_color[1], width=6)
    # draw guidelines
    drawer.line((0, height//2, width, height//2), fill='lightgrey', width=1)
    drawer.line((width//2,  0, width//2, height), fill='lightgrey', width=1)
    drawer.circle((width//2, height//2), radius=4, fill='grey') # target dot
    # draw actual
    drawer.circle((yaw, pitch), radius=8, fill=fill_color[0])
    # draw bbox
    drawer.rectangle((0, 0, width-1, height-1), fill=None, outline='black', width=1)
    

    # stitch
    padding = 24
    combined = PIL.Image.new('RGB', (512+padding*3, 768-image_tz.width+padding*3), color='lightgrey')
    combined.paste(image_txy, (padding*1, padding*1))
    combined.paste(image_tz,  (image_txy.width+padding*2, padding*1))
    combined.paste(image_rr,  (padding*1, image_txy.height+padding*2))
    combined.paste(image_ryp, (image_rr.width+padding*2, image_txy.height+padding*2))

    drawer  = PIL.ImageDraw.Draw(combined)

    try:
        # font = PIL.ImageFont.truetype("arial.tff", 16)
        font = PIL.ImageFont.load_default()

        # label txy
        mask = PIL.Image.new('RGBA', (image_txy.height+padding*2, image_txy.width+padding*2), (0, 0, 0, 0))
        drawer = PIL.ImageDraw.Draw(mask)
        drawer.text((padding, padding//2), f'{-xy_bbox_size}', font=font, fill='black', anchor='lt')
        drawer.text((image_txy.height//2+padding, padding//2), f'Y', font=font, fill='black', anchor='mt')
        drawer.text((image_txy.height+padding, padding//2), f'{xy_bbox_size}', font=font, fill='black', anchor='rt')
        mask = mask.rotate(90)
        drawer = PIL.ImageDraw.Draw(mask)
        drawer.text((padding, padding//2), f'{-xy_bbox_size}', font=font, fill='black', anchor='lt')
        drawer.text((image_txy.width//2+padding, padding//2), f'X', font=font, fill='black', anchor='mt')
        drawer.text((image_txy.width+padding, padding//2), f'{xy_bbox_size}', font=font, fill='black', anchor='rt')
        combined.paste(mask, (0, 0), mask)

        # label tz
        mask = PIL.Image.new('RGBA', (image_tz.height+padding*2, image_tz.width+padding*2), (0, 0, 0, 0))
        drawer = PIL.ImageDraw.Draw(mask)
        drawer.text((padding, padding//2), f'{-z_bbox_size}', font=font, fill='black', anchor='lt')
        drawer.text((image_txy.height//2+padding, padding//2), f'Z', font=font, fill='black', anchor='mt')
        drawer.text((image_txy.height+padding, padding//2), f'{z_bbox_size}', font=font, fill='black', anchor='rt')
        mask = mask.rotate(90, expand=True)
        combined.paste(mask, (image_txy.width+padding, 0), mask)

        # label rr
        mask = PIL.Image.new('RGBA', (image_rr.width+padding*2, image_rr.height+padding*2), (0, 0, 0, 0))
        drawer = PIL.ImageDraw.Draw(mask)
        drawer.text((padding, padding//2), f'{-r_bbox_size}', font=font, fill='black', anchor='lt')
        drawer.text((image_rr.width//2+padding, padding//2), f'Roll', font=font, fill='black', anchor='mt')
        drawer.text((image_rr.width+padding, padding//2), f'{r_bbox_size}', font=font, fill='black', anchor='rt')
        combined.paste(mask, (0, image_txy.height+padding), mask)

        # label ryp
        mask = PIL.Image.new('RGBA', (image_ryp.height+padding*2, image_ryp.width+padding*2), (0, 0, 0, 0))
        drawer = PIL.ImageDraw.Draw(mask)
        drawer.text((padding, padding//2), f'{-yp_bbox_size}', font=font, fill='black', anchor='lt')
        drawer.text((image_ryp.height//2+padding, padding//2), f'Pitch', font=font, fill='black', anchor='mt')
        drawer.text((image_ryp.height+padding, padding//2), f'{yp_bbox_size}', font=font, fill='black', anchor='rt')
        # mask.rotate(90, expand=True, resample=PIL.Image.Resampling.BICUBIC)
        mask = mask.rotate(90)
        drawer = PIL.ImageDraw.Draw(mask)
        drawer.text((padding, padding//2), f'{-yp_bbox_size}', font=font, fill='black', anchor='lt')
        drawer.text((image_ryp.width//2+padding, padding//2), f'Yaw', font=font, fill='black', anchor='mt')
        drawer.text((image_ryp.width+padding, padding//2), f'{yp_bbox_size}', font=font, fill='black', anchor='rt')
        combined.paste(mask, (image_rr.width+padding, image_txy.height+padding), mask)


    except Exception as e:
        print("---------------------------------------------------------------------------------------------------------------")
        print("If you crash here with 'cannot open resource', it's probably because you don't have libraqm installed. Do that.")
        print("---------------------------------------------------------------------------------------------------------------")
        raise e

    return combined


if __name__ == '__main__':
    a = np.eye(4)
    b = np.eye(4)
    # b[:3, 3] = np.array([0, 0, 0])
    b[:3, 3] = np.array([-1.1, 6, -1.3])
    # b[:3, :3] = Rotation.from_euler('XYZ', [0, 0, 0], degrees=True).as_matrix()
    b[:3, :3] = Rotation.from_euler('XYZ', [30, -80, 70], degrees=True).as_matrix()

    trans, euler = relative_transform(a, b)
    # print(trans, euler)

    # reconstructed = apply_transform(b,
    # [1.1, -6, 1.3],
    # [-110, 125, 95],
    # )

    # trans, euler = relative_transform(a, reconstructed)
    # print(trans, euler)

    img = render_pose(trans, euler)
    img.save('out.png')