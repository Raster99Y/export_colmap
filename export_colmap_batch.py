import os
import shutil
import struct
import math
import time
from pathlib import Path
from PySide2 import QtCore, QtGui, QtWidgets


# Checking compatibility
compatible_major_version = "2.1"
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))


f32 = lambda x: bytes(struct.pack("f", x))
d64 = lambda x: bytes(struct.pack("d", x))
u8  = lambda x: x.to_bytes(1, "little", signed=(x < 0))
u32 = lambda x: x.to_bytes(4, "little", signed=(x < 0))
u64 = lambda x: x.to_bytes(8, "little", signed=(x < 0))
bstr = lambda x: bytes((x + "\0"), "utf-8")

def matrix_to_quat(m):
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if (tr > 0):
        s = 2 * math.sqrt(tr + 1)
        return Metashape.Vector([(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s])
    if (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        s = 2 * math.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2])
        return Metashape.Vector([0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s])
    if (m[1, 1] > m[2, 2]):
        s = 2 * math.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2])
        return Metashape.Vector([(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s])
    else:
        s = 2 * math.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1])
        return Metashape.Vector([(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s])

def get_camera_name(cam):
    return os.path.basename(cam.photo.path)

def clean_dir(folder, confirm_deletion):
    if os.path.exists(folder):
        if confirm_deletion:
            ok = Metashape.app.getBool('Folder "' + folder + '" will be deleted.\nAre you sure you want to continue?')
            if not ok:
                return False
        shutil.rmtree(folder)
    os.mkdir(folder)
    return True

def build_dir_structure(folder, confirm_deletion):
    if not os.path.exists(folder):
        os.makedirs(folder)
    if not clean_dir(folder + "images/", confirm_deletion):
        return False
    if not clean_dir(folder + "sparse/", confirm_deletion):
        return False
    os.makedirs(folder + "sparse/0/")
    return True

def get_chunk_dirs(folder, params):
    doc = Metashape.app.document
    chunk_name_stats = {}
    chunk_names = {}

    initial_chunk_selected = doc.chunk.selected
    doc.chunk.selected = True

    for chunk in doc.chunks:
        if not params.all_chunks and not chunk.selected:
            continue
        label = chunk.label
        i = chunk_name_stats[label] = chunk_name_stats.get(label, 0)
        while True:
            name = folder + label + ("" if i == 0 else "_" + str(i)) + "/"
            i += 1
            if name not in chunk_names.values():
                chunk_names[chunk.key] = name
                chunk_name_stats[label] = i
                break

    doc.chunk.selected = initial_chunk_selected

    existed = [name for name in chunk_names.values() if os.path.exists(name)]
    if len(existed) > 0:
        ok = Metashape.app.getBool('These folders will be deleted:\n"' + '"\n"'.join(existed) + '"\nAre you sure you want to continue?')
        if not ok:
            return {}
    for name in existed:
        shutil.rmtree(name)
    return chunk_names

def calib_valid(calib, point):
    reproj = calib.project(calib.unproject(point))
    if not reproj:
        return False
    return (reproj - point).norm() < 1.0

def get_valid_calib_region(calib):
    w = calib.width
    h = calib.height

    left = math.floor(calib.cx + w / 2)
    right = math.floor(calib.cx + w / 2)
    top = math.floor(calib.cy + h / 2)
    bottom = math.floor(calib.cy + h / 2)

    left_set = False
    right_set = False
    top_set = False
    bottom_set = False

    max_dim = max(w, h)
    max_tan = math.hypot(w, h) / calib.f

    step_x = 1
    step_y = 1

    if w > h:
        step_y /= min(1.2, (w / h))
    else:
        step_x /= min(1.2, (h / w))

    for r in range(max_dim):
        if left_set and top_set and right_set and bottom_set:
            break

        next_top = top if top_set else math.floor(calib.cy + h / 2 - r * step_y)
        next_bottom = bottom if bottom_set else math.floor(calib.cy + h / 2 + r * step_y)
        next_left = left if left_set else math.floor(calib.cx + w / 2 - r * step_x)
        next_right = right if right_set else math.floor(calib.cx + w / 2 + r * step_x)

        next_top = max(next_top, 0)
        next_left = max(next_left, 0)
        next_right = min(next_right, w - 1)
        next_bottom = min(next_bottom, h - 1)

        for v in range(2):
            for u in range(2):

                if u == 0 and left_set:
                    continue
                if v == 0 and top_set:
                    continue
                if u == 1 and right_set:
                    continue
                if v == 1 and bottom_set:
                    continue

                corner = Metashape.Vector([next_right if u else next_left, next_bottom if v else next_top])
                corner.x += 0.5
                corner.y += 0.5

                step = Metashape.Vector([step_x if u else -step_x, step_y if v else -step_y])

                prev_corner = Metashape.Vector(corner)
                prev_corner -= step

                pt = calib.unproject(corner)

                pt = Metashape.Vector([pt.x / pt.z, pt.y / pt.z])

                prev_pt = calib.unproject(prev_corner)
                prev_pt = Metashape.Vector([prev_pt.x / prev_pt.z, prev_pt.y / prev_pt.z])

                dif = pt - prev_pt

                if (pt.norm() > max_tan or dif * step <= 0 or not calib_valid(calib, corner)):
                    if u:
                        right_set = True
                    else:
                        left_set = True

                    if v:
                        bottom_set = True
                    else:
                        top_set = True

        if not left_set:
            left = next_left
        if not top_set:
            top = next_top
        if not right_set:
            right = next_right
        if not bottom_set:
            bottom = next_bottom

    right += 1
    bottom += 1

    new_w = right - left
    new_h = bottom - top

    border = math.ceil(0.01 * min(new_w, new_h))

    if left_set:
        left += border
    if right_set:
        right -= border
    if top_set:
        top += border
    if bottom_set:
        bottom -= border

    new_w = right - left
    new_h = bottom - top

    if (new_w != w or new_h != h):
        print("Cropped initial calibration due to high distortion", str(w) + "x" + str(h), "->", str(new_w) + "x" + str(new_h))

    return (top, right, bottom, left)

def rotate_vector(vec, axis, angle):
    axis = axis.normalized()
    collinear = axis * (vec * axis)
    orthogonal0 = vec - collinear
    orthogonal1 = Metashape.Vector.cross(axis, orthogonal0)
    return collinear + orthogonal0 * math.cos(angle) + orthogonal1 * math.sin(angle)

def axis_magnitude_rotation(axis):
    angle = axis.norm()
    axis = axis.normalized()
    x = Metashape.Vector((1, 0, 0))
    y = Metashape.Vector((0, 1, 0))
    z = Metashape.Vector((0, 0, 1))
    return Metashape.Matrix((rotate_vector(x, axis, -angle), rotate_vector(y, axis, -angle), rotate_vector(z, axis, -angle)))

def compute_size(top, right, bottom, left, T1):
    T1_inv = T1.inv()

    tl = T1_inv.mulp(Metashape.Vector([left, top, 1]))
    tr = T1_inv.mulp(Metashape.Vector([right, top, 1]))
    bl = T1_inv.mulp(Metashape.Vector([left, bottom, 1]))
    br = T1_inv.mulp(Metashape.Vector([right, bottom, 1]))

    halfwl = min(-tl.x / tl.z, -bl.x / bl.z)
    halfwr = min(tr.x / tr.z, br.x / br.z)
    halfht = min(-tr.y / tr.z, -tl.y / tl.z)
    halfhb = min(br.y / br.z, bl.y / bl.z)

    return (halfht, halfwr, halfhb, halfwl)

def compute_undistorted_calib(sensor, zero_cxy):
    border = 0 # in pixels, can be increased if black margins are on the undistorted images

    if sensor.type != Metashape.Sensor.Type.Frame and sensor.type != Metashape.Sensor.Type.Fisheye:
        return (Metashape.Calibration(), Metashape.Matrix.Diag([1, 1, 1, 1]))

    calib_initial = sensor.calibration
    w = calib_initial.width
    h = calib_initial.height
    f = calib_initial.f

    (reg_top, reg_right, reg_bottom, reg_left) = get_valid_calib_region(calib_initial)

    left = -float("inf")
    right = float("inf")
    top = -float("inf")
    bottom = float("inf")

    for i in range(reg_top, reg_bottom):
        im_pt = Metashape.Vector([reg_left + 0.5, i + 0.5])
        if (calib_valid(calib_initial, im_pt)):
            pt = calib_initial.unproject(im_pt)
            left = max(left, pt.x / pt.z)

        im_pt = Metashape.Vector([reg_right - 0.5, i + 0.5])
        if (calib_valid(calib_initial, im_pt)):
            pt = calib_initial.unproject(im_pt)
            right = min(right, pt.x / pt.z)

    for i in range(reg_left, reg_right):
        im_pt = Metashape.Vector([i + 0.5, reg_top + 0.5])
        if (calib_valid(calib_initial, im_pt)):
            pt = calib_initial.unproject(im_pt)
            top = max(top, pt.y / pt.z)

        im_pt = Metashape.Vector([i + 0.5, reg_bottom - 0.5])
        if (calib_valid(calib_initial, im_pt)):
            pt = calib_initial.unproject(im_pt)
            bottom = min(bottom, pt.y / pt.z)

    if right <= left or bottom <= top:
        return (Metashape.Calibration(), Metashape.Matrix.Diag([1, 1, 1, 1]))

    T1 = Metashape.Matrix.Diag([1, 1, 1, 1])
    if zero_cxy:
        left_ang = math.atan(left)
        right_ang = math.atan(right)
        top_ang = math.atan(top)
        bottom_ang = math.atan(bottom)

        rotation_vec = Metashape.Vector([math.tan((left_ang + right_ang) / 2), math.tan((top_ang + bottom_ang) / 2), 1]).normalized()
        rotation_vec = Metashape.Vector.cross(Metashape.Vector((0, 0, 1)), rotation_vec)
        T1 = Metashape.Matrix.Rotation(axis_magnitude_rotation(rotation_vec))

    (halfht, halfwr, halfhb, halfwl) = compute_size(top, right, bottom, left, T1)

    halfht = math.floor(f * halfht)
    halfwr = math.floor(f * halfwr)
    halfhb = math.floor(f * halfhb)
    halfwl = math.floor(f * halfwl)

    halfw = min(halfwl, halfwr)
    halfh = min(halfht, halfhb)

    if zero_cxy:
        halfwl = halfw
        halfwr = halfw
        halfht = halfh
        halfhb = halfh

    max_dim = max(w, h)

    calib = Metashape.Calibration()
    calib.f = f
    calib.width = min(math.floor(max_dim * 1.2), math.floor(halfwl + halfwr) - 2 * border)
    calib.height = min(math.floor(max_dim * 1.2), math.floor(halfht + halfhb) - 2 * border)
    calib.cx = halfwl - (halfwl + halfwr) / 2
    calib.cy = halfht - (halfht + halfhb) / 2

    return (calib, T1)

def check_undistorted_calib(sensor, calib, T1):
    border = 0 # in pixels, can be increased if black margins are on the undistorted images

    calib_initial = sensor.calibration
    w = calib.width
    h = calib.height

    left = float("inf")
    right = -float("inf")
    top = float("inf")
    bottom = -float("inf")

    for i in range(h):
        pt = calib_initial.project(T1.mulp(calib.unproject(Metashape.Vector([0.5, i + 0.5]))))
        left = min(left, pt.x)
        pt = calib_initial.project(T1.mulp(calib.unproject(Metashape.Vector([w - 0.5, i + 0.5]))))
        right = max(right, pt.x)
    for i in range(w):
        pt = calib_initial.project(T1.mulp(calib.unproject(Metashape.Vector([i + 0.5, 0.5]))))
        top = min(top, pt.y)
        pt = calib_initial.project(T1.mulp(calib.unproject(Metashape.Vector([i + 0.5, h - 0.5]))))
        bottom = max(bottom, pt.y)

    print(left, right, top, bottom)
    if (left < 0.5 or calib_initial.width - 0.5 < right or top < 0.5 or calib_initial.height - 0.5 < bottom):
        print("!!! Wrong undistorted calib")
    else:
        print("Ok:")

def get_coord_transform(frame, use_localframe):
    if not use_localframe:
        return frame.transform.matrix
    if not frame.region:
        print("Null region, using world crs instead of local")
        return frame.transform.matrix
    fr_to_gc  = frame.transform.matrix
    gc_to_loc = frame.crs.localframe(fr_to_gc.mulp(frame.region.center))
    fr_to_loc = gc_to_loc * fr_to_gc
    return (Metashape.Matrix.Translation(-fr_to_loc.mulp(frame.region.center)) * fr_to_loc)

def compute_undistorted_calibs(frame, zero_cxy):
    calibs = {} # { sensor_key: ( sensor, undistorted calibration, undistorted camera transform ) }
    for sensor in frame.sensors:
        (calib, T1) = compute_undistorted_calib(sensor, zero_cxy)

        if (calib.width == 0 or calib.height == 0):
            continue
        calibs[sensor.key] = (sensor, calib, T1)
        #check_undistorted_calib(sensor, calib, T1)

    print("Calibrations:")
    for (s_key, (sensor, calib, T1)) in calibs.items():
        print(sensor.key, calib.f, calib.width, calib.height, calib.cx, calib.cy)

    return calibs

def get_calibs(camera, calibs):
    s_key = camera.sensor.key
    if s_key not in calibs:
        cause = "unsupported" if (camera.sensor.type != Metashape.Sensor.Type.Frame and camera.sensor.type != Metashape.Sensor.Type.Fisheye) else "cropped"
        print("Camera " + camera.label + " (key = " + str(camera.key) + ") has " + cause + " sensor (key = " + str(s_key) + ")")
        return (None, None, None)
    return (calibs[s_key][0].calibration, calibs[s_key][1], calibs[s_key][2])


def save_undistorted_images(params, frame, folder, calibs):
    print("Exporting images.")
    folder = folder + "images/"
    T = Metashape.Matrix.Diag([1, 1, 1, 1])

    cnt = 0
    valid_cameras = []
    for cam in frame.cameras:
        if cam.transform is None or cam.sensor is None or not cam.enabled:
            continue
        if cam.sensor.key not in calibs:
            continue
        (calib0, calib1, T1) = get_calibs(cam, calibs)
        if calib0 is None:
            continue
        valid_cameras.append((cam, calib0, calib1, T1))
    
    total_cameras = len(valid_cameras)
    if total_cameras > 0:
        progress_controller.update_progress(0, total_cameras, "导出去畸变图像...")
    
    for i, (cam, calib0, calib1, T1) in enumerate(valid_cameras):
        if progress_controller.is_cancelled():
            return
            
        img = cam.image().warp(calib0, T, calib1, T1)
        name = get_camera_name(cam)
        ext = os.path.splitext(name)[1]
        if ext.lower() in [".jpg", ".jpeg"]:
            c = Metashape.ImageCompression()
            c.jpeg_quality = params.image_quality
            img.save(folder + name, c)
        else:
            img.save(folder + name)
        cnt += 1
        
        progress_controller.update_progress(i + 1, total_cameras, f"导出图像 {i+1}/{total_cameras}")
        
    print("Undistorted", cnt, "cameras")

def save_undistorted_masks(params, frame, folder, calibs):
    print("Exporting masks.")
    folder = folder + "masks/"
    if not clean_dir(folder, params.confirm_deletion):
        print("Masks folder already exists, aborting.")
        return False

    T = Metashape.Matrix.Diag([1, 1, 1, 1])

    cnt = 0
    valid_cameras = []
    for cam in frame.cameras:
        if cam.transform is None or cam.sensor is None or not cam.enabled:
            continue
        if cam.sensor.key not in calibs:
            continue
        (calib0, calib1, T1) = get_calibs(cam, calibs)
        if calib0 is None:
            continue
        if not cam.mask:
            continue
        valid_cameras.append((cam, calib0, calib1, T1))
    
    total_cameras = len(valid_cameras)
    if total_cameras > 0:
        progress_controller.update_progress(0, total_cameras, "导出去畸变掩膜...")
    
    for i, (cam, calib0, calib1, T1) in enumerate(valid_cameras):
        if progress_controller.is_cancelled():
            return
            
        mask = cam.mask.image().warp(calib0, T, calib1, T1)
        mask = mask.convert("L")
        name = get_camera_name(cam)
        mask.save(str(Path(folder + name).with_suffix('.png')))
        cnt += 1
        
        progress_controller.update_progress(i + 1, total_cameras, f"导出掩膜 {i+1}/{total_cameras}")
        
    print("Undistorted", cnt, "masks")

def save_cameras(params, folder, calibs):
    use_pinhole_model = params.use_pinhole_model
    with open(folder + "sparse/0/cameras.bin", "wb") as fout:
        fout.write(u64(len(calibs)))
        for (s_key, (sensor, calib, T1)) in calibs.items():
            fout.write(u32(s_key))
            fout.write(u32(1 if use_pinhole_model else 0))
            fout.write(u64(calib.width))
            fout.write(u64(calib.height))
            fout.write(d64(calib.f))
            if use_pinhole_model:
                fout.write(d64(calib.f))
            fout.write(d64(calib.cx + calib.width * 0.5))
            fout.write(d64(calib.cy + calib.height * 0.5))
    print("Saved", len(calibs), "calibrations")

def save_camera_params(params, folder, calibs):
    """保存相机参数及畸变参数到txt文件"""
    # 创建一个相机参数文件
    with open(folder + "sparse/0/cameras_params.txt", "w") as f:
        f.write("# camera_id width height fx fy cx cy k1 k2 k3 k4 p1 p2 b1 b2\n")
        print(list(calibs.values())[0])
        for (s_key, (sensor, calib_undist, sub_camera_transform)) in calibs.items():
            # 获取原始校准
            calib_orig = sensor.calibration
            
            # 获取相机内参
            f_val = calib_orig.f
            cx = calib_orig.cx + calib_orig.width * 0.5
            cy = calib_orig.cy + calib_orig.height * 0.5
            width = calib_orig.width
            height = calib_orig.height
            
            # 获取畸变参数
            k1 = calib_orig.k1
            k2 = calib_orig.k2
            k3 = calib_orig.k3
            k4 = calib_orig.k4 if hasattr(calib_orig, 'k4') else 0.0
            p1 = calib_orig.p1
            p2 = calib_orig.p2
            # 获取薄棱镜畸变系数 (Metashape中为b1, b2)
            b1 = getattr(calib_orig, 'b1', 0.0)
            b2 = getattr(calib_orig, 'b2', 0.0)
            
            # 写入参数
            f.write(f"{s_key} {width} {height} {f_val} {f_val} {cx} {cy} {k1} {k2} {k3} {k4} {p1} {p2} {b1} {b2}\n")
    
    print("已保存", len(calibs), "个相机的参数到cameras_params.txt")


class ExportSceneParams():
    def __init__(self):
        # default values for parameters
        self.all_chunks = False
        self.all_frames = False
        self.zero_cxy = False
        self.use_localframe = True
        self.image_quality = 90
        self.export_images = False
        self.export_masks = False
        self.confirm_deletion = True
        self.use_pinhole_model = True
        self.only_good = True
        self.batch_size = 500000
        self.export_ply = True  # 添加PLY导出选项

    def log(self):
        print("All chunks:", self.all_chunks)
        print("All frames:", self.all_frames)
        print("Zero cx and cy:", self.zero_cxy)
        print("Use local coordinate frame:", self.use_localframe)
        print("Image quality:", self.image_quality)
        print("Export images:", self.export_images)
        print("Export masks:", self.export_masks)
        print("Confirm deletion:", self.confirm_deletion)
        print("Using pinhole model instead of simple_pinhole:", self.use_pinhole_model)
        print("Using only uncropped projections:", self.only_good)
        print("Batch size:", self.batch_size)
        print("Export PLY:", self.export_ply)  # 添加PLY导出日志


# ================================
# 进度控制系统
# ================================

class ProgressController:
    """进度控制器，支持进度显示和取消操作"""
    
    def __init__(self):
        self.cancelled = False
        self.current_step = ""
        self.progress_value = 0
        self.max_value = 100
        self.callback = None
        self._initialized = False
        
    def set_callback(self, callback):
        """设置进度回调函数"""
        self.callback = callback
        self._initialized = True
        
    def update_progress(self, value, max_val=None, step_name=""):
        """更新进度"""
        if not self._initialized:
            return
            
        if max_val is not None:
            self.max_value = max_val
        self.progress_value = value
        self.current_step = step_name
        
        if self.callback:
            self.callback(value, self.max_value, step_name)
            
        QtWidgets.QApplication.processEvents()
        
    def cancel(self):
        """取消操作"""
        self.cancelled = True
        
    def is_cancelled(self):
        """检查是否已取消"""
        return self.cancelled
        
    def reset(self):
        """重置控制器状态"""
        self.cancelled = False
        self.current_step = ""
        self.progress_value = 0
        self.max_value = 100
        self._initialized = False

# 全局进度控制器
progress_controller = ProgressController()

def export_for_gaussian_splatting(params = ExportSceneParams(), progress = QtWidgets.QProgressBar()):
    log_result = lambda x: print("", x, "-----------------------------------", sep="\n")
    
    params.log()

    folder = Metashape.app.getExistingDirectory("Output folder")
    if len(folder) == 0:
        log_result("No chosen folder")
        return False
    folder = folder + "/"
    print(folder)

    chunk_dirs = get_chunk_dirs(folder, params)
    if len(chunk_dirs) == 0:
        log_result("Aborted")
        return False

    chunk_num = len(chunk_dirs)
    for chunk_id, (chunk_key, chunk_folder) in enumerate(chunk_dirs.items()):
        if progress_controller.is_cancelled():
            return False
            
        chunk = [ck for ck in Metashape.app.document.chunks if ck.key == chunk_key]
        if (len(chunk) != 1):
            print("Chunk not found, key =", chunk_key)
            continue
        chunk = chunk[0]

        frame_num = len(chunk.frames) if params.all_frames else 1
        frame_cnt = 0

        for frame_id, frame in enumerate(chunk.frames):
            if progress_controller.is_cancelled():
                return False
                
            if not frame.tie_points:
                continue
            if not params.all_frames and not (frame == chunk.frame):
                continue
            frame_cnt += 1

            folder_path = chunk_folder + ("" if frame_num == 1 else "frame_" + str(frame_id).zfill(6) + "/")
            print("\n" + folder_path)

            if not build_dir_structure(folder_path, params.confirm_deletion):
                log_result("Aborted")
                return False

            progress_controller.update_progress(0, 100, f"处理 Chunk {chunk_id+1}/{chunk_num}")
            calibs = compute_undistorted_calibs(frame, params.zero_cxy)
            
            if progress_controller.is_cancelled():
                return False

            if params.export_images:
                save_undistorted_images(params, frame, folder_path, calibs)
            if params.export_masks:
                save_undistorted_masks(params, frame, folder_path, calibs)
                
            if progress_controller.is_cancelled():
                return False
                
            progress_controller.update_progress(30, 100, "保存相机参数...")
            save_camera_params(params, folder_path, calibs)

            if progress_controller.is_cancelled():
                return False
                
            # 使用分批处理替代原来的get_filtered_track_structure
            progress_controller.update_progress(40, 100, "分批处理数据结构...")
            batch_result = get_filtered_track_structure_batch(frame, folder_path, calibs, params.batch_size)  # 使用参数中的批次大小
            if batch_result is None:  # 用户取消
                return False
            
            batch_files, temp_dir, total_points = batch_result
            
            if progress_controller.is_cancelled():
                cleanup_temp_files(temp_dir)
                return False
                
            # 分批保存数据
            save_images_batch(params, frame, folder_path, calibs, batch_files, temp_dir)
            
            if progress_controller.is_cancelled():
                cleanup_temp_files(temp_dir)
                return False
                
            save_points_batch(params, frame, folder_path, calibs, batch_files, temp_dir)
            
            # 清理临时文件
            cleanup_temp_files(temp_dir)
            
            if progress_controller.is_cancelled():
                return False

    progress_controller.update_progress(100, 100, "导出完成！")
    log_result("Done")
    return True

class ProgressDialog(QtWidgets.QDialog):
    """进度对话框"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("导出进度")
        self.setModal(True)
        self.setMinimumSize(500, 150)
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowCloseButtonHint)
        self._export_completed = False
        self._user_cancelled = False
        self.setup_ui()
        
    def setup_ui(self):
        layout = QtWidgets.QVBoxLayout()
        
        self.step_label = QtWidgets.QLabel("准备开始...")
        self.step_label.setAlignment(QtCore.Qt.AlignCenter)
        self.step_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(self.step_label)
        
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        layout.addWidget(self.progress_bar)
        
        self.detail_label = QtWidgets.QLabel("等待开始...")
        self.detail_label.setAlignment(QtCore.Qt.AlignCenter)
        self.detail_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(self.detail_label)
        
        self.time_label = QtWidgets.QLabel("")
        self.time_label.setAlignment(QtCore.Qt.AlignCenter)
        self.time_label.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(self.time_label)
        
        button_layout = QtWidgets.QHBoxLayout()
        self.cancel_btn = QtWidgets.QPushButton("取消")
        self.cancel_btn.setFixedSize(100, 30)
        self.cancel_btn.clicked.connect(self.cancel_export)
        button_layout.addStretch()
        button_layout.addWidget(self.cancel_btn)
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        progress_controller.set_callback(self.update_progress)
        self.start_time = time.time()
        
    def update_progress(self, value, max_value, step_name):
        """更新进度显示"""
        self.progress_bar.setMaximum(max_value)
        self.progress_bar.setValue(value)
        
        if step_name:
            self.step_label.setText(step_name)
            
        if max_value > 0:
            percentage = (value / max_value) * 100
            self.detail_label.setText(f"进度: {value}/{max_value} ({percentage:.1f}%)")
        
        elapsed_time = time.time() - self.start_time
        if value > 0 and max_value > 0 and not self._export_completed:
            estimated_total = elapsed_time * max_value / value
            remaining_time = estimated_total - elapsed_time
            self.time_label.setText(f"已用: {elapsed_time:.0f}s | 预计剩余: {remaining_time:.0f}s")
        else:
            self.time_label.setText(f"已用时: {elapsed_time:.0f}s")
        
        if value >= max_value and step_name and ("完成" in step_name or "导出完成" in step_name):
            self._export_completed = True
            self.cancel_btn.setText("关闭")
            self.cancel_btn.setStyleSheet("background-color: #28a745; color: white; border: none; border-radius: 5px;")
            
        self.repaint()
        
    def cancel_export(self):
        if self._export_completed:
            self.accept()
            return
        
        reply = QtWidgets.QMessageBox.question(
            self, "确认取消", 
            "确定要取消导出吗？\n已处理的数据可能会丢失。",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )
        
        if reply == QtWidgets.QMessageBox.Yes:
            self._user_cancelled = True
            progress_controller.cancel()
            self.step_label.setText("正在取消...")
            self.cancel_btn.setEnabled(False)
            QtCore.QTimer.singleShot(1000, self.accept)

    def closeEvent(self, event):
        if self._export_completed or self._user_cancelled:
            event.accept()
            return
        event.ignore()
        self.cancel_export()
        
    def mark_completed(self):
        self._export_completed = True
        self.cancel_btn.setText("关闭")
        self.cancel_btn.setEnabled(True)
        self.cancel_btn.setStyleSheet("background-color: #28a745; color: white; border: none; border-radius: 5px;")

class CollapsibleGroupBox(QtWidgets.QGroupBox):
    def __init__(self, parent = None):
        QtWidgets.QGroupBox.__init__(self, parent)
        self.toggled.connect(self.onCheckedChanged)
        self.maxHeight = self.maximumHeight()

    def onCheckedChanged(self, is_on):
        if not is_on:
            self.oldSize = self.size()

        for child in self.children():
            if isinstance(child, QtWidgets.QWidget):
                child.setVisible(is_on)

        if is_on:
            self.setMaximumHeight(self.maxHeight)
            self.resize(self.oldSize)
        else:
            self.maxHeight = self.maximumHeight()
            self.setMaximumHeight(QtGui.QFontMetrics(self.font()).height() + 4)


class ExportSceneGUI(QtWidgets.QDialog):

    def run_export(self):
        """运行导出"""
        params = ExportSceneParams()
        params.all_chunks = self.radioBtn_allC.isChecked()
        params.all_frames = self.radioBtn_allF.isChecked()
        params.zero_cxy = self.zcxyBox.isChecked()
        params.use_localframe = self.locFrameBox.isChecked()
        params.image_quality = self.imgQualSpBox.value()
        params.export_images = self.expImagesBox.isChecked()
        params.export_masks = self.expMasksBox.isChecked()
        params.batch_size = self.batchSizeSpBox.value()
        params.export_ply = self.expPlyBox.isChecked()  # 添加PLY导出参数获取
        
        # 重置进度控制器
        progress_controller.reset()
        
        # 显示进度对话框
        progress_dialog = ProgressDialog(self)
        start_time = time.time()
        
        try:
            progress_dialog.show()
            QtWidgets.QApplication.processEvents()
            
            # 执行导出
            success = export_for_gaussian_splatting(params, progress_dialog.progress_bar)
            
            elapsed_time = time.time() - start_time
            
            # 标记完成并更新界面
            if success and not progress_controller.is_cancelled():
                progress_dialog.mark_completed()
                progress_dialog.step_label.setText("导出完成！")
                progress_dialog.detail_label.setText(f"成功完成，耗时 {elapsed_time:.1f} 秒")
                progress_dialog.time_label.setText("可以关闭此对话框")
                
                QtWidgets.QMessageBox.information(
                    progress_dialog, "导出完成", 
                    f"导出成功完成！\n耗时: {elapsed_time:.1f} 秒"
                )
                
            elif progress_controller.is_cancelled():
                progress_dialog.step_label.setText("导出已取消")
                progress_dialog.detail_label.setText("用户取消了导出操作")
                progress_dialog.mark_completed()
                QtWidgets.QMessageBox.information(progress_dialog, "导出已取消", "导出操作已被用户取消。")
                
            else:
                progress_dialog.step_label.setText("导出失败")
                progress_dialog.detail_label.setText("导出过程中遇到错误")
                progress_dialog.mark_completed()
                QtWidgets.QMessageBox.warning(progress_dialog, "导出失败", "导出过程中遇到错误，请检查控制台日志。")
            
            progress_dialog.exec_()
            
        except Exception as e:
            progress_dialog.mark_completed()
            QtWidgets.QMessageBox.critical(progress_dialog, "导出错误", f"导出过程中发生异常:\n{str(e)}")
            progress_dialog.close()
        finally:
            if progress_dialog.isVisible():
                progress_dialog.close()
            self.close()

    def __init__(self, parent):
        QtWidgets.QDialog.__init__(self, parent)
        self.setWindowTitle("Export COLMAP Project - With Progress")
        self.setMinimumSize(450, 500)

        defaults = ExportSceneParams()

        # 标题
        title_label = QtWidgets.QLabel("COLMAP导出")
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px; color: #2E86AB;")
        
        # 说明
        info_label = QtWidgets.QLabel("支持实时进度显示、可随时取消")
        info_label.setAlignment(QtCore.Qt.AlignCenter)
        info_label.setStyleSheet("color: #666; margin-bottom: 15px;")

        # 基本控件
        self.btnQuit = QtWidgets.QPushButton("退出")
        self.btnQuit.setFixedSize(100,25)

        self.btnP1 = QtWidgets.QPushButton("开始导出")
        self.btnP1.setFixedSize(100,25)
        self.btnP1.setStyleSheet("background-color: #28a745; color: white; border: none; border-radius: 5px; font-weight: bold;")

        self.chnkTxt = QtWidgets.QLabel()
        self.chnkTxt.setText("Chunks:")
        self.chnkTxt.setFixedSize(100, 25)

        self.frmsTxt = QtWidgets.QLabel()
        self.frmsTxt.setText("Frames:")
        self.frmsTxt.setFixedSize(100, 25)

        self.chunk_group = QtWidgets.QButtonGroup()
        self.radioBtn_allC = QtWidgets.QRadioButton("all chunks")
        self.radioBtn_selC = QtWidgets.QRadioButton("selected")
        self.chunk_group.addButton(self.radioBtn_selC)
        self.chunk_group.addButton(self.radioBtn_allC)
        self.radioBtn_allC.setChecked(defaults.all_chunks)
        self.radioBtn_selC.setChecked(not defaults.all_chunks)

        self.frames_group = QtWidgets.QButtonGroup()
        self.radioBtn_allF = QtWidgets.QRadioButton("all frames")
        self.radioBtn_selF = QtWidgets.QRadioButton("active")
        self.frames_group.addButton(self.radioBtn_selF)
        self.frames_group.addButton(self.radioBtn_allF)
        self.radioBtn_allF.setChecked(defaults.all_frames)
        self.radioBtn_selF.setChecked(not defaults.all_frames)

        self.zcxyTxt = QtWidgets.QLabel()
        self.zcxyTxt.setText("Enforce zero cx, cy")
        self.zcxyTxt.setFixedSize(100, 25)

        self.zcxyBox = QtWidgets.QCheckBox()
        self.zcxyBox.setChecked(defaults.zero_cxy)

        self.locFrameTxt = QtWidgets.QLabel()
        self.locFrameTxt.setText("Use localframe")
        self.locFrameTxt.setFixedSize(100, 25)

        self.locFrameBox = QtWidgets.QCheckBox()
        self.locFrameBox.setChecked(defaults.use_localframe)

        self.imgQualTxt = QtWidgets.QLabel()
        self.imgQualTxt.setText("Image quality")
        self.imgQualTxt.setFixedSize(100, 25)

        self.imgQualSpBox = QtWidgets.QSpinBox()
        self.imgQualSpBox.setMinimum(0)
        self.imgQualSpBox.setMaximum(100)
        self.imgQualSpBox.setValue(defaults.image_quality)

        self.expImagesTxt = QtWidgets.QLabel()
        self.expImagesTxt.setText("Export images")
        self.expImagesTxt.setFixedSize(100, 25)

        self.expImagesBox = QtWidgets.QCheckBox()
        self.expImagesBox.setChecked(defaults.export_images)

        self.expMasksTxt = QtWidgets.QLabel()
        self.expMasksTxt.setText("Export masks")
        self.expMasksTxt.setFixedSize(100, 25)

        self.expMasksBox = QtWidgets.QCheckBox()
        self.expMasksBox.setChecked(defaults.export_masks)

        # 添加PLY导出控件
        self.expPlyTxt = QtWidgets.QLabel()
        self.expPlyTxt.setText("Export PLY")
        self.expPlyTxt.setFixedSize(100, 25)

        self.expPlyBox = QtWidgets.QCheckBox()
        self.expPlyBox.setChecked(defaults.export_ply)

        # 添加批次大小控件
        self.batchSizeTxt = QtWidgets.QLabel()
        self.batchSizeTxt.setText("Batch size")
        self.batchSizeTxt.setFixedSize(100, 25)

        self.batchSizeSpBox = QtWidgets.QSpinBox()
        self.batchSizeSpBox.setMinimum(10000)
        self.batchSizeSpBox.setMaximum(100000000)
        self.batchSizeSpBox.setSingleStep(10000)
        self.batchSizeSpBox.setValue(defaults.batch_size)

        # 添加工具提示
        batchSizeToolTip = "批处理大小 - 控制内存使用\n较大值: 更快但占用更多内存\n较小值: 更慢但占用更少内存\n推荐值: 500000"
        self.batchSizeTxt.setToolTip(batchSizeToolTip)
        self.batchSizeSpBox.setToolTip(batchSizeToolTip)

        expMasksToolTip = "You can enable export of the undistorted masks"
        self.expMasksTxt.setToolTip(expMasksToolTip)
        self.expMasksBox.setToolTip(expMasksToolTip)

        expPlyToolTip = "导出PLY点云文件\n可用于CloudCompare、MeshLab等软件查看\n包含3D坐标和颜色信息"
        self.expPlyTxt.setToolTip(expPlyToolTip)
        self.expPlyBox.setToolTip(expPlyToolTip)

        general_layout = QtWidgets.QGridLayout()
        general_layout.setSpacing(9)
        general_layout.addWidget(self.chnkTxt, 1, 0)
        general_layout.addWidget(self.radioBtn_allC, 1, 1)
        general_layout.addWidget(self.radioBtn_selC, 1, 2)
        general_layout.addWidget(self.frmsTxt, 2, 0)
        general_layout.addWidget(self.radioBtn_allF, 2, 1)
        general_layout.addWidget(self.radioBtn_selF, 2, 2)
        general_layout.addWidget(self.zcxyTxt, 3, 0)
        general_layout.addWidget(self.zcxyBox, 3, 1)
        general_layout.addWidget(self.locFrameTxt, 4, 0)
        general_layout.addWidget(self.locFrameBox, 4, 1)
        general_layout.addWidget(self.imgQualTxt, 5, 0)
        general_layout.addWidget(self.imgQualSpBox, 5, 1, 1, 2)

        advanced_layout = QtWidgets.QGridLayout()
        advanced_layout.setSpacing(9)
        advanced_layout.addWidget(self.expImagesTxt, 0, 0)
        advanced_layout.addWidget(self.expImagesBox, 0, 1)
        advanced_layout.setSpacing(9)
        advanced_layout.addWidget(self.expMasksTxt, 1, 0)
        advanced_layout.addWidget(self.expMasksBox, 1, 1)
        advanced_layout.addWidget(self.expPlyTxt, 2, 0)  # 添加PLY导出控件
        advanced_layout.addWidget(self.expPlyBox, 2, 1)
        advanced_layout.addWidget(self.batchSizeTxt, 3, 0)  # 调整批次大小位置
        advanced_layout.addWidget(self.batchSizeSpBox, 3, 1, 1, 2)

        self.gbGeneral = QtWidgets.QGroupBox()
        self.gbGeneral.setLayout(general_layout)
        self.gbGeneral.setTitle("基本设置")

        self.gbAdvanced = CollapsibleGroupBox()
        self.gbAdvanced.setLayout(advanced_layout)
        self.gbAdvanced.setTitle("高级设置")
        self.gbAdvanced.setCheckable(True)
        self.gbAdvanced.setChecked(False)
        self.gbAdvanced.toggled.connect(lambda: QtCore.QTimer.singleShot(20, lambda: self.adjustSize()))

        # 信息框
        memory_info = QtWidgets.QLabel()
        memory_info.setText("💡 注意:\n• 若要生成PLY格式文件，请在高级设置中勾选export PLY\n• 可在高级设置中调节批次大小，默认为500000\n• 可用最小批次为10000，最大批次为100000000")
        memory_info.setStyleSheet("""
            background-color: #e8f4fd; border: 1px solid #bee5eb; border-radius: 5px;
            padding: 10px; margin: 5px 0px; font-size: 11px; color: #0c5460;
        """)

        layout = QtWidgets.QGridLayout()
        layout.addWidget(title_label, 0, 0, 1, 3)
        layout.addWidget(info_label, 1, 0, 1, 3)
        layout.addWidget(self.gbGeneral, 2, 0, 1, 3)
        layout.addWidget(self.gbAdvanced, 3, 0, 1, 3)
        layout.addWidget(memory_info, 4, 0, 1, 3)
        layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.MinimumExpanding), 5, 0, 1, 3)
        layout.addWidget(self.btnP1, 6, 1)
        layout.addWidget(self.btnQuit, 6, 2)
        self.setLayout(layout)

        self.buttons = [self.btnP1, self.btnQuit, self.radioBtn_allC, self.radioBtn_selC, self.radioBtn_allF, self.radioBtn_selF, self.zcxyBox, self.locFrameBox, self.imgQualSpBox, self.expImagesBox, self.expMasksBox, self.expPlyBox, self.batchSizeSpBox]  # 添加PLY导出控件到按钮列表

        QtCore.QObject.connect(self.btnP1, QtCore.SIGNAL("clicked()"), self.run_export)
        QtCore.QObject.connect(self.btnQuit, QtCore.SIGNAL("clicked()"), self, QtCore.SLOT("reject()"))

        self.exec()


def export_for_gaussian_splatting_gui():
    global app
    app = QtWidgets.QApplication.instance()
    parent = app.activeWindow()
    
    # 检查是否有打开的项目
    doc = Metashape.app.document
    if not doc or not hasattr(doc, 'chunks') or not doc.chunks:
        QtWidgets.QMessageBox.warning(None, "警告", "请先打开一个包含数据的Metashape项目")
        return
    
    # 检查是否有tie_points
    has_tie_points = False
    for chunk in doc.chunks:
        for frame in chunk.frames:
            if frame.tie_points and frame.tie_points.points:
                has_tie_points = True
                break
        if has_tie_points:
            break
    
    if not has_tie_points:
        QtWidgets.QMessageBox.warning(None, "警告", "项目中没有找到tie points数据。\n请先完成稀疏重建。")
        return
    
    try:
        dlg = ExportSceneGUI(parent)
    except Exception as e:
        QtWidgets.QMessageBox.critical(None, "错误", f"启动界面时发生错误:\n{str(e)}")

# 使用不同的标签避免冲突
label = "Scripts/Export COLMAP - Batch Processing (with PLY)"
Metashape.app.addMenuItem(label, export_for_gaussian_splatting_gui)
print("To execute this script press {}".format(label))

def cleanup_temp_files(temp_dir):
    """清理临时文件"""
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print("Cleaned up temporary files")
    except Exception as e:
        print(f"Warning: Could not clean temp files: {e}")

def get_filtered_track_structure_batch(frame, folder, calibs, batch_size=500000):
    """分批处理跟踪点结构，减少内存占用"""
    tie_points = frame.tie_points
    total_points = len(tie_points.points)
    
    # 临时文件路径
    temp_dir = folder + "temp_batch/"
    os.makedirs(temp_dir, exist_ok=True)
    
    progress_controller.update_progress(0, total_points, "分批处理跟踪点...")
    
    # 分批处理点
    batch_files = []
    for start_idx in range(0, total_points, batch_size):
        if progress_controller.is_cancelled():
            return None, None, None
            
        end_idx = min(start_idx + batch_size, total_points)
        
        # 修复：使用索引遍历而不是切片
        batch_points = []
        for i in range(start_idx, end_idx):
            batch_points.append(tie_points.points[i])
        
        # 处理当前批次
        batch_tracks, batch_images = process_point_batch(
            batch_points, start_idx, frame, calibs, tie_points
        )
        
        # 保存批次结果
        batch_file = temp_dir + f"batch_{start_idx}_{end_idx}.pkl"
        save_batch_data(batch_file, batch_tracks, batch_images)
        batch_files.append(batch_file)
        
        progress_controller.update_progress(end_idx, total_points, 
            f"处理跟踪点 {end_idx}/{total_points}")
        
        # 清理内存
        del batch_tracks, batch_images, batch_points
        import gc
        gc.collect()
    
    return batch_files, temp_dir, total_points

def process_point_batch(batch_points, start_idx, frame, calibs, tie_points):
    """处理单个批次的点"""
    tracks = {}
    images = {}
    
    # 构建当前批次的track映射
    for i, pt in enumerate(batch_points):
        track_id = pt.track_id
        if track_id not in tracks:
            tracks[track_id] = [[], [], []]
        tracks[track_id][0].append(start_idx + i)
    
    # 处理相机投影
    for cam in frame.cameras:
        if cam.transform is None or cam.sensor is None or not cam.enabled:
            continue
        (calib0, calib1, T1) = get_calibs(cam, calibs)
        if calib0 is None:
            continue

        T1_inv = T1.inv()
        # 修复：只保存相机key，不保存Camera对象
        camera_entry = [cam.key, [], []]  # 改为只保存cam.key
        
        projections = tie_points.projections[cam]
        for proj in projections:
            track_id = proj.track_id
            if track_id not in tracks:
                continue

            try:
                pt = calib1.project(T1_inv.mulp(calib0.unproject(proj.coord)))
                good = False
                if pt is not None:
                    good = (0 <= pt.x < calib1.width and 0 <= pt.y < calib1.height)

                place = 1 if good else 2
                pos = len(camera_entry[place])
                
                # 修复：只保存基本数据，不保存Metashape对象
                pt_data = (float(pt.x), float(pt.y)) if pt else (0.0, 0.0)
                camera_entry[place].append((pt_data, float(proj.size), int(track_id)))
                tracks[track_id][place].append((cam.key, pos))
            except Exception as e:
                print(f"Error processing projection for camera {cam.key}: {e}")
                continue

        if cam.key not in images:
            images[cam.key] = camera_entry
        else:
            # 合并投影数据
            existing = images[cam.key]
            existing[1].extend(camera_entry[1])
            existing[2].extend(camera_entry[2])
    
    return tracks, images

def save_batch_data(filename, tracks, images):
    """保存批次数据到文件"""
    import pickle
    with open(filename, 'wb') as f:
        pickle.dump({
            'tracks': tracks,
            'images': images
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_batch_data(filename):
    """从文件加载批次数据"""
    import pickle
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data['tracks'], data['images']

def merge_camera_data_from_batches(batch_files, frame):
    """从批次文件中合并相机数据"""
    merged_images = {}
    
    for batch_file in batch_files:
        if progress_controller.is_cancelled():
            return {}
            
        _, batch_images = load_batch_data(batch_file)
        
        for cam_key, camera_data in batch_images.items():
            if cam_key not in merged_images:
                merged_images[cam_key] = camera_data
            else:
                # 合并投影数据
                existing = merged_images[cam_key]
                existing[1].extend(camera_data[1])
                existing[2].extend(camera_data[2])
        
        # 立即清理
        del batch_images
        import gc
        gc.collect()
    
    return merged_images

def save_images_batch(params, frame, folder, calibs, batch_files, temp_dir):
    """优化后的图像数据保存"""
    only_good = params.only_good
    T_shift = get_coord_transform(frame, params.use_localframe)
    
    # 获取所有相机信息
    camera_basic_info = {}
    for cam in frame.cameras:
        if cam.transform is None or cam.sensor is None or not cam.enabled:
            continue
        (calib0, calib1, T1) = get_calibs(cam, calibs)
        if calib0 is None:
            continue
        camera_basic_info[cam.key] = (cam, calib0, calib1, T1)
    
    progress_controller.update_progress(0, len(batch_files), "正在处理相机数据...")
    
    # 预构建所有投影数据到内存（只保存必要信息）
    camera_projections = {cam_key: [] for cam_key in camera_basic_info.keys()}
    
    for i, batch_file in enumerate(batch_files):
        if progress_controller.is_cancelled():
            return
            
        tracks, batch_images = load_batch_data(batch_file)
        
        # 快速收集投影数据
        for cam_key, camera_data in batch_images.items():
            if cam_key in camera_projections:
                projections_to_add = camera_data[1] if only_good else camera_data[1] + camera_data[2]
                # 直接添加元组数据，避免复杂对象
                for pt_data, size, track_id in projections_to_add:
                    camera_projections[cam_key].append((pt_data[0], pt_data[1], track_id))
        
        del tracks, batch_images
        import gc
        gc.collect()
        
        progress_controller.update_progress(i + 1, len(batch_files), 
            f"收集投影数据 {i + 1}/{len(batch_files)}")
    
    # 快速写入相机数据
    progress_controller.update_progress(0, len(camera_basic_info), "写入相机位姿...")
    
    with open(folder + "sparse/0/images.bin", "wb") as fout:
        fout.write(u64(len(camera_basic_info)))
        
        for i, (cam_key, (camera, calib0, calib1, T1)) in enumerate(camera_basic_info.items()):
            if progress_controller.is_cancelled():
                return
                
            # 预计算变换矩阵
            transform = T_shift * camera.transform * T1
            R = transform.rotation().inv()
            T = -1 * (R * transform.translation())
            Q = matrix_to_quat(R)
            
            # 批量写入相机信息
            fout.write(u32(cam_key))
            fout.write(d64(Q.w))
            fout.write(d64(Q.x))
            fout.write(d64(Q.y))
            fout.write(d64(Q.z))
            fout.write(d64(T.x))
            fout.write(d64(T.y))
            fout.write(d64(T.z))
            fout.write(u32(camera.sensor.key))
            fout.write(bstr(get_camera_name(camera)))
            
            # 批量写入投影数据
            projections = camera_projections[cam_key]
            fout.write(u64(len(projections)))
            
            # 高效写入投影
            for pt_x, pt_y, track_id in projections:
                fout.write(d64(pt_x))
                fout.write(d64(pt_y))
                fout.write(u64(track_id))
            
            if i % 10 == 0:  # 减少进度更新频率
                progress_controller.update_progress(i + 1, len(camera_basic_info), 
                    f"保存相机 {i + 1}/{len(camera_basic_info)}")
    
    print(f"High-speed saved {len(camera_basic_info)} cameras")

def save_points_batch(params, frame, folder, calibs, batch_files, temp_dir):
    """优化后的点云数据保存 - 一次遍历完成所有操作"""
    only_good = params.only_good
    T = get_coord_transform(frame, params.use_localframe)
    
    progress_controller.update_progress(0, len(batch_files), "正在处理点云数据...")
    
    # 预计算相机投影长度映射
    camera_good_proj_lengths = {}
    
    # 一次遍历完成：统计、索引构建、二进制写入、PLY写入
    binary_temp_file = temp_dir + "points3d_temp.bin"
    ply_temp_file = temp_dir + "ply_temp.txt" if params.export_ply else None
    
    total_points_written = 0
    
    with open(binary_temp_file, 'wb') as bin_temp:
        # 逐批次处理
        for batch_idx, batch_file in enumerate(batch_files):
            if progress_controller.is_cancelled():
                return
                
            tracks, batch_images = load_batch_data(batch_file)
            
            # 更新相机投影长度映射
            for cam_key, camera_data in batch_images.items():
                if cam_key not in camera_good_proj_lengths:
                    camera_good_proj_lengths[cam_key] = 0
                camera_good_proj_lengths[cam_key] += len(camera_data[1])
            
            # 批量处理当前批次的所有有效点
            batch_points_data = []  # 收集当前批次的点数据
            
            for track_id, track_data in tracks.items():
                points, good_prjs, bad_prjs = track_data
                
                if len(points) == 0:
                    continue
                
                try:
                    # 预处理3D点数据
                    point = frame.tie_points.points[points[0]]
                    pt = T * point.coord
                    track = frame.tie_points.tracks[track_id]
                    
                    # 预处理投影数据
                    projections = []
                    for (camera_key, proj_idx) in good_prjs:
                        projections.append((camera_key, proj_idx))
                    
                    if not only_good:
                        for (camera_key, proj_idx) in bad_prjs:
                            # 使用当前已知的长度
                            adjusted_idx = proj_idx + camera_good_proj_lengths.get(camera_key, 0)
                            projections.append((camera_key, adjusted_idx))
                    
                    batch_points_data.append({
                        'track_id': track_id,
                        'position': (pt.x, pt.y, pt.z),
                        'color': (track.color[0], track.color[1], track.color[2]),
                        'projections': projections
                    })
                    
                except Exception as e:
                    print(f"Error preprocessing point {track_id}: {e}")
                    continue
            
            # 批量写入二进制数据
            for point_data in batch_points_data:
                bin_temp.write(u64(point_data['track_id']))
                bin_temp.write(d64(point_data['position'][0]))
                bin_temp.write(d64(point_data['position'][1]))
                bin_temp.write(d64(point_data['position'][2]))
                bin_temp.write(u8(point_data['color'][0]))
                bin_temp.write(u8(point_data['color'][1]))
                bin_temp.write(u8(point_data['color'][2]))
                bin_temp.write(d64(0))  # error
                
                # 写入投影数据
                bin_temp.write(u64(len(point_data['projections'])))
                for camera_key, proj_idx in point_data['projections']:
                    bin_temp.write(u32(camera_key))
                    bin_temp.write(u32(proj_idx))
                
                total_points_written += 1
            
            # 清理当前批次
            del tracks, batch_images, batch_points_data
            import gc
            gc.collect()
            
            progress_controller.update_progress(batch_idx + 1, len(batch_files), 
                f"处理批次 {batch_idx + 1}/{len(batch_files)} - 已写入 {total_points_written} 点")
    
    # 快速复制到最终文件
    progress_controller.update_progress(0, 100, "生成最终文件...")
    
    # 写入最终二进制文件
    with open(folder + "sparse/0/points3D.bin", "wb") as final_bin:
        final_bin.write(u64(total_points_written))
        with open(binary_temp_file, 'rb') as temp_bin:
            # 大块复制
            while True:
                chunk = temp_bin.read(1048576)  # 1MB 块
                if not chunk:
                    break
                final_bin.write(chunk)
    
    # 生成PLY文件 - 修正版本，直接从数据生成
    if params.export_ply and total_points_written > 0:
        generate_ply_from_batch_files(batch_files, frame, T, folder + "sparse/0/points3D.ply", total_points_written, only_good)
    
    # 清理临时文件
    try:
        os.remove(binary_temp_file)
    except:
        pass
    
    progress_controller.update_progress(100, 100, "点云数据保存完成")
    print(f"High-speed saved {total_points_written} points with optimized processing")

def generate_ply_from_batch_files(batch_files, frame, transform_matrix, ply_output_path, expected_points, only_good):
    """直接从批次文件生成PLY - 确保文件完整性"""
    progress_controller.update_progress(0, len(batch_files), "生成PLY文件...")
    
    # 预计算相机投影长度映射（用于bad投影索引调整）
    camera_good_proj_lengths = {}
    for batch_file in batch_files:
        _, batch_images = load_batch_data(batch_file)
        for cam_key, camera_data in batch_images.items():
            if cam_key not in camera_good_proj_lengths:
                camera_good_proj_lengths[cam_key] = 0
            camera_good_proj_lengths[cam_key] += len(camera_data[1])
        del batch_images
    
    actual_points_written = 0
    
    # 首先统计实际的点数
    for batch_file in batch_files:
        tracks, _ = load_batch_data(batch_file)
        for track_id, track_data in tracks.items():
            points, good_prjs, bad_prjs = track_data
            if len(points) > 0:
                actual_points_written += 1
        del tracks
    
    print(f"Expected {expected_points} points, found {actual_points_written} points for PLY")
    
    with open(ply_output_path, 'w', encoding='utf-8', newline='\n') as ply_file:
        # 写入PLY文件头 - 使用实际点数
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write(f"element vertex {actual_points_written}\n")
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("property uchar red\n")
        ply_file.write("property uchar green\n")
        ply_file.write("property uchar blue\n")
        ply_file.write("end_header\n")
        
        points_written = 0
        
        # 写入点数据
        for batch_idx, batch_file in enumerate(batch_files):
            if progress_controller.is_cancelled():
                return
                
            tracks, _ = load_batch_data(batch_file)
            
            for track_id, track_data in tracks.items():
                points, good_prjs, bad_prjs = track_data
                
                if len(points) == 0:
                    continue
                
                try:
                    # 获取3D点数据
                    point = frame.tie_points.points[points[0]]
                    pt = transform_matrix * point.coord
                    track = frame.tie_points.tracks[track_id]
                    
                    # 写入PLY格式的点数据 - 确保数据完整
                    ply_file.write(f"{pt.x:.6f} {pt.y:.6f} {pt.z:.6f} {track.color[0]} {track.color[1]} {track.color[2]}\n")
                    points_written += 1
                        
                except Exception as e:
                    print(f"Error writing PLY point {track_id}: {e}")
                    continue
            
            del tracks
            import gc
            gc.collect()
            
            progress_controller.update_progress(batch_idx + 1, len(batch_files), 
                f"写入PLY批次 {batch_idx + 1}/{len(batch_files)} - 已写入 {points_written} 点")
    
    # 验证PLY文件完整性
    if points_written != actual_points_written:
        print(f"Warning: PLY point count mismatch. Expected {actual_points_written}, written {points_written}")
    
    progress_controller.update_progress(len(batch_files), len(batch_files), "PLY文件生成完成")
    print(f"Generated PLY file with {points_written} points: {ply_output_path}")