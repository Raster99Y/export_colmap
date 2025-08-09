import os
import shutil
import struct
import math
import time
from pathlib import Path
from PySide2 import QtCore, QtGui, QtWidgets

# 版本兼容性检查
compatible_major_version = "2.1"
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))

# 数据类型转换函数
f32 = lambda x: bytes(struct.pack("f", x))
d64 = lambda x: bytes(struct.pack("d", x))
u8  = lambda x: x.to_bytes(1, "little", signed=(x < 0))
u32 = lambda x: x.to_bytes(4, "little", signed=(x < 0))
u64 = lambda x: x.to_bytes(8, "little", signed=(x < 0))
bstr = lambda x: bytes((x + "\0"), "utf-8")

# ================================
# 进度控制系统
# ================================

class ProgressController:
    """全局进度控制器，支持进度显示和取消操作"""
    
    def __init__(self):
        self.cancelled = False
        self.current_step = ""
        self.progress_value = 0
        self.max_value = 100
        self.callback = None
        
    def set_callback(self, callback):
        """设置进度回调函数"""
        self.callback = callback
        
    def update_progress(self, value, max_val=None, step_name=""):
        """更新进度"""
        if max_val is not None:
            self.max_value = max_val
        self.progress_value = value
        self.current_step = step_name
        
        if self.callback:
            self.callback(value, self.max_value, step_name)
            
        # 保持界面响应
        QtWidgets.QApplication.processEvents()
        
    def cancel(self):
        """取消操作"""
        self.cancelled = True
        
    def is_cancelled(self):
        """检查是否已取消"""
        return self.cancelled

# 全局进度控制器
progress_controller = ProgressController()

# ================================
# 数学和几何计算函数
# ================================

def matrix_to_quat(m):
    """将旋转矩阵转换为四元数"""
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

def get_coord_transform(frame, use_localframe):
    """获取坐标变换矩阵"""
    if not use_localframe:
        return frame.transform.matrix
    if not frame.region:
        print("Null region, using world crs instead of local")
        return frame.transform.matrix
    fr_to_gc  = frame.transform.matrix
    gc_to_loc = frame.crs.localframe(fr_to_gc.mulp(frame.region.center))
    fr_to_loc = gc_to_loc * fr_to_gc
    return (Metashape.Matrix.Translation(-fr_to_loc.mulp(frame.region.center)) * fr_to_loc)

# ================================
# 文件和目录管理
# ================================

def get_camera_name(cam):
    """获取相机文件名"""
    return os.path.basename(cam.photo.path)

def clean_dir(folder, confirm_deletion):
    """清理目录"""
    if os.path.exists(folder):
        if confirm_deletion:
            ok = Metashape.app.getBool('Folder "' + folder + '" will be deleted.\nAre you sure you want to continue?')
            if not ok:
                return False
        shutil.rmtree(folder)
    os.mkdir(folder)
    return True

def build_dir_structure(folder, confirm_deletion):
    """构建输出目录结构"""
    if not os.path.exists(folder):
        os.makedirs(folder)
    if not clean_dir(folder + "images/", confirm_deletion):
        return False
    if not clean_dir(folder + "sparse/", confirm_deletion):
        return False
    os.makedirs(folder + "sparse/0/")
    return True

def get_chunk_dirs(folder, params):
    """获取chunk输出目录"""
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

    if not params.all_frames and len(chunk_names) == 1:
        return {chunk_key:folder for chunk_key in chunk_names}

    existed = [name for name in chunk_names.values() if os.path.exists(name)]
    if len(existed) > 0:
        ok = Metashape.app.getBool('These folders will be deleted:\n"' + '"\n"'.join(existed) + '"\nAre you sure you want to continue?')
        if not ok:
            return {}
    for name in existed:
        shutil.rmtree(name)
    return chunk_names

# ================================
# 相机校准相关函数
# ================================

def calib_valid(calib, point):
    """检查校准点是否有效"""
    reproj = calib.project(calib.unproject(point))
    if not reproj:
        return False
    return (reproj - point).norm() < 1.0

def get_valid_calib_region(calib):
    """获取有效校准区域"""
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
    """旋转向量"""
    axis = axis.normalized()
    collinear = axis * (vec * axis)
    orthogonal0 = vec - collinear
    orthogonal1 = Metashape.Vector.cross(axis, orthogonal0)
    return collinear + orthogonal0 * math.cos(angle) + orthogonal1 * math.sin(angle)

def axis_magnitude_rotation(axis):
    """轴角旋转"""
    angle = axis.norm()
    axis = axis.normalized()
    x = Metashape.Vector((1, 0, 0))
    y = Metashape.Vector((0, 1, 0))
    z = Metashape.Vector((0, 0, 1))
    return Metashape.Matrix((rotate_vector(x, axis, -angle), rotate_vector(y, axis, -angle), rotate_vector(z, axis, -angle)))

def compute_size(top, right, bottom, left, T1):
    """计算尺寸"""
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
    """计算去畸变校准"""
    border = 0

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

def compute_undistorted_calibs(frame, zero_cxy):
    """计算所有去畸变校准，支持进度显示"""
    calibs = {}
    sensors = frame.sensors
    total_sensors = len(sensors)
    
    progress_controller.update_progress(0, total_sensors, "计算相机校准...")
    
    for i, sensor in enumerate(sensors):
        if progress_controller.is_cancelled():
            return {}
            
        (calib, T1) = compute_undistorted_calib(sensor, zero_cxy)

        if (calib.width == 0 or calib.height == 0):
            continue
        calibs[sensor.key] = (sensor, calib, T1)
        
        progress_controller.update_progress(i + 1, total_sensors, f"计算校准 {i+1}/{total_sensors}")

    print("Calibrations computed:", len(calibs))
    return calibs

def get_calibs(camera, calibs):
    """获取相机校准参数"""
    s_key = camera.sensor.key
    if s_key not in calibs:
        return (None, None, None)
    return (calibs[s_key][0].calibration, calibs[s_key][1], calibs[s_key][2])

# ================================
# 图像和掩膜导出
# ================================

def save_undistorted_images(params, frame, folder, calibs):
    """保存去畸变图像，支持进度显示"""
    if not params.export_images:
        return
        
    print("Exporting undistorted images...")
    folder = folder + "images/"
    T = Metashape.Matrix.Diag([1, 1, 1, 1])

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
        
        progress_controller.update_progress(i + 1, total_cameras, f"导出图像 {i+1}/{total_cameras}")
        
    print(f"Exported {total_cameras} undistorted images")

def save_undistorted_masks(params, frame, folder, calibs):
    """保存去畸变掩膜，支持进度显示"""
    if not params.export_masks:
        return
        
    print("Exporting undistorted masks...")
    folder = folder + "masks/"
    if not clean_dir(folder, params.confirm_deletion):
        return

    T = Metashape.Matrix.Diag([1, 1, 1, 1])

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
    progress_controller.update_progress(0, total_cameras, "导出去畸变掩膜...")

    for i, (cam, calib0, calib1, T1) in enumerate(valid_cameras):
        if progress_controller.is_cancelled():
            return
            
        mask = cam.mask.image().warp(calib0, T, calib1, T1)
        mask = mask.convert("L")
        name = get_camera_name(cam)
        mask.save(str(Path(folder + name).with_suffix('.png')))
        
        progress_controller.update_progress(i + 1, total_cameras, f"导出掩膜 {i+1}/{total_cameras}")
        
    print(f"Exported {total_cameras} undistorted masks")

# ================================
# COLMAP文件保存
# ================================

def save_cameras(params, folder, calibs):
    """保存相机参数到COLMAP格式"""
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
    print(f"Saved {len(calibs)} camera calibrations")

def save_camera_params(params, folder, calibs):
    """保存详细相机参数到txt文件"""
    with open(folder + "sparse/0/cameras_params.txt", "w") as f:
        f.write("# camera_id width height fx fy cx cy k1 k2 k3 k4 p1 p2 b1 b2\n")
        for (s_key, (sensor, calib_undist, sub_camera_transform)) in calibs.items():
            calib_orig = sensor.calibration
            
            f_val = calib_orig.f
            cx = calib_orig.cx + calib_orig.width * 0.5
            cy = calib_orig.cy + calib_orig.height * 0.5
            width = calib_orig.width
            height = calib_orig.height
            
            k1 = calib_orig.k1
            k2 = calib_orig.k2
            k3 = calib_orig.k3
            k4 = calib_orig.k4 if hasattr(calib_orig, 'k4') else 0.0
            p1 = calib_orig.p1
            p2 = calib_orig.p2
            b1 = getattr(calib_orig, 'b1', 0.0)
            b2 = getattr(calib_orig, 'b2', 0.0)
            
            f.write(f"{s_key} {width} {height} {f_val} {f_val} {cx} {cy} {k1} {k2} {k3} {k4} {p1} {p2} {b1} {b2}\n")
    
    print(f"Saved detailed parameters for {len(calibs)} cameras")

# ================================
# 内存优化的流式处理核心
# ================================

def create_camera_projection_index(frame, calibs):
    """创建轻量级相机索引"""
    camera_info = {}
    valid_cameras = []
    
    for cam in frame.cameras:
        if cam.transform is None or cam.sensor is None or not cam.enabled:
            continue
        (calib0, calib1, T1) = get_calibs(cam, calibs)
        if calib0 is None:
            continue
        valid_cameras.append(cam)
        
    total_cameras = len(valid_cameras)
    progress_controller.update_progress(0, total_cameras, "创建相机索引...")
    
    for i, cam in enumerate(valid_cameras):
        if progress_controller.is_cancelled():
            return {}
            
        (calib0, calib1, T1) = get_calibs(cam, calibs)
        
        camera_info[cam.key] = {
            'camera': cam,
            'calib0': calib0,
            'calib1': calib1, 
            'T1': T1,
            'T1_inv': T1.inv()
        }
        
        progress_controller.update_progress(i + 1, total_cameras, f"索引相机 {i+1}/{total_cameras}")
    
    print(f"Created index for {len(camera_info)} cameras")
    return camera_info

def save_images_streaming(params, frame, folder, calibs, camera_info):
    """流式保存相机位姿数据"""
    only_good = params.only_good
    T_shift = get_coord_transform(frame, params.use_localframe)
    
    total_cameras = len(camera_info)
    progress_controller.update_progress(0, total_cameras, "写入相机位姿...")
    
    with open(folder + "sparse/0/images.bin", "wb") as fout:
        fout.write(u64(len(camera_info)))
        
        for i, (cam_key, cam_info) in enumerate(camera_info.items()):
            if progress_controller.is_cancelled():
                return
                
            camera = cam_info['camera']
            (calib0, calib1, T1) = (cam_info['calib0'], cam_info['calib1'], cam_info['T1'])
            
            # 计算相机位姿
            transform = T_shift * camera.transform * T1
            R = transform.rotation().inv()
            T = -1 * (R * transform.translation())
            Q = matrix_to_quat(R)
            
            # 写入相机信息
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
            
            # 计算和写入投影数据
            projections = frame.tie_points.projections[camera]
            valid_projections = []
            
            for proj in projections:
                pt_proj = cam_info['calib1'].project(
                    cam_info['T1_inv'].mulp(cam_info['calib0'].unproject(proj.coord))
                )
                
                good = (pt_proj is not None and 
                       0 <= pt_proj.x < cam_info['calib1'].width and 
                       0 <= pt_proj.y < cam_info['calib1'].height)
                
                if only_good and not good:
                    continue
                    
                valid_projections.append((pt_proj, proj.track_id))
            
            # 写入投影数量和数据
            fout.write(u64(len(valid_projections)))

            for pt_proj, track_id in valid_projections:
                fout.write(d64(pt_proj.x if pt_proj else 0))
                fout.write(d64(pt_proj.y if pt_proj else 0))
                fout.write(u64(track_id))
            
            progress_controller.update_progress(i + 1, total_cameras, f"写入相机 {i+1}/{total_cameras}")
    
    print(f"Saved {len(camera_info)} camera poses")

def save_points_streaming(params, frame, folder, calibs, camera_info, batch_size=1000000):
    """流式处理点云数据 - 内存优化核心"""
    only_good = params.only_good
    T = get_coord_transform(frame, params.use_localframe)
    tie_points = frame.tie_points
    
    # 创建临时文件夹
    temp_folder = folder + "temp_batches/"
    os.makedirs(temp_folder, exist_ok=True)
    
    batch_files = []
    batch_num = 0
    current_batch = []
    total_points_processed = 0
    valid_points_count = 0
    
    total_points = len(tie_points.points)
    progress_controller.update_progress(0, total_points, "流式处理点云...")
    
    try:
        # 流式处理每个点
        for point_idx, point in enumerate(tie_points.points):
            if progress_controller.is_cancelled():
                return 0
                
            track_id = point.track_id
            
            # 查找该点的投影
            projections = []
            for cam_key, cam_info in camera_info.items():
                camera = cam_info['camera']
                cam_projections = tie_points.projections[camera]
                
                for proj_idx, proj in enumerate(cam_projections):
                    if proj.track_id == track_id:
                        # 计算投影坐标
                        pt_proj = cam_info['calib1'].project(
                            cam_info['T1_inv'].mulp(cam_info['calib0'].unproject(proj.coord))
                        )
                        
                        good = (pt_proj is not None and 
                               0 <= pt_proj.x < cam_info['calib1'].width and 
                               0 <= pt_proj.y < cam_info['calib1'].height)
                        
                        if only_good and not good:
                            continue
                            
                        projections.append((cam_key, proj_idx))
            
            # 如果没有有效投影，跳过这个点
            if not projections:
                continue
                
            # 计算3D坐标
            pt = T * point.coord
            track = tie_points.tracks[track_id]
            
            # 添加到当前批次
            current_batch.append({
                'track_id': track_id,
                'position': (pt.x, pt.y, pt.z),
                'color': (track.color[0], track.color[1], track.color[2]),
                'projections': projections
            })
            
            valid_points_count += 1
            total_points_processed += 1
            
            # 更新进度
            if total_points_processed % 1000 == 0:
                progress_controller.update_progress(
                    total_points_processed, 
                    total_points, 
                    f"处理点云 {total_points_processed}/{total_points} (有效: {valid_points_count})"
                )
            
            # 批次满了，保存并清空
            if len(current_batch) >= batch_size:
                if progress_controller.is_cancelled():
                    return 0
                    
                batch_file = temp_folder + f"batch_{batch_num:04d}.bin"
                save_batch_to_file(current_batch, batch_file)
                batch_files.append(batch_file)
                
                print(f"Saved batch {batch_num + 1}, {len(current_batch)} points")
                current_batch = []  # 释放内存
                batch_num += 1
        
        # 保存最后一个批次
        if current_batch and not progress_controller.is_cancelled():
            batch_file = temp_folder + f"batch_{batch_num:04d}.bin"
            save_batch_to_file(current_batch, batch_file)
            batch_files.append(batch_file)
            print(f"Saved final batch, {len(current_batch)} points")
        
        if progress_controller.is_cancelled():
            return 0
        
        # 合并批次文件
        progress_controller.update_progress(0, 3, "合并批次文件...")
        merge_batches_to_colmap_binary(batch_files, folder + "sparse/0/points3D.bin", valid_points_count)
        
        if progress_controller.is_cancelled():
            return 0
            
        # 生成PLY文件
        progress_controller.update_progress(1, 3, "生成PLY文件...")
        convert_to_ply(batch_files, folder + "sparse/0/points3D.ply")
        
        progress_controller.update_progress(3, 3, f"完成！处理了 {valid_points_count} 个点")
        print(f"Stream processing completed: {valid_points_count} points")
        return valid_points_count
        
    finally:
        # 清理临时文件
        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)

def save_batch_to_file(batch_data, filename):
    """保存批次数据到文件"""
    with open(filename, 'wb') as f:
        f.write(u64(len(batch_data)))
        
        for point_data in batch_data:
            f.write(u64(point_data['track_id']))
            f.write(d64(point_data['position'][0]))
            f.write(d64(point_data['position'][1]))
            f.write(d64(point_data['position'][2]))
            f.write(u8(point_data['color'][0]))
            f.write(u8(point_data['color'][1]))
            f.write(u8(point_data['color'][2]))
            f.write(d64(0))  # 误差占位符
            f.write(u64(len(point_data['projections'])))
            for camera_key, proj_idx in point_data['projections']:
                f.write(u32(camera_key))
                f.write(u32(proj_idx))

def merge_batches_to_colmap_binary(batch_files, output_file, total_points):
    """合并批次文件到COLMAP二进制格式"""
    with open(output_file, "wb") as fout:
        fout.write(u64(total_points))
        
        for batch_file in batch_files:
            with open(batch_file, 'rb') as fin:
                fin.read(8)  # 跳过批次点数
                while True:
                    chunk = fin.read(8192)
                    if not chunk:
                        break
                    fout.write(chunk)

def convert_to_ply(batch_files, ply_filename):
    """将批次文件转换为PLY格式"""
    total_points = 0
    
    # 计算总点数
    for batch_file in batch_files:
        with open(batch_file, 'rb') as f:
            points_in_batch = struct.unpack('<Q', f.read(8))[0]
            total_points += points_in_batch
    
    # 写入PLY文件
    with open(ply_filename, 'w') as ply_file:
        # PLY头部
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write(f"element vertex {total_points}\n")
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("property uchar red\n")
        ply_file.write("property uchar green\n")
        ply_file.write("property uchar blue\n")
        ply_file.write("end_header\n")
        
        # 写入点数据
        for batch_file in batch_files:
            with open(batch_file, 'rb') as f:
                points_in_batch = struct.unpack('<Q', f.read(8))[0]
                
                for _ in range(points_in_batch):
                    f.read(8)  # 跳过track ID
                    
                    x = struct.unpack('<d', f.read(8))[0]
                    y = struct.unpack('<d', f.read(8))[0]
                    z = struct.unpack('<d', f.read(8))[0]
                    
                    r = struct.unpack('<B', f.read(1))[0]
                    g = struct.unpack('<B', f.read(1))[0]
                    b = struct.unpack('<B', f.read(1))[0]
                    
                    f.read(8)  # 跳过误差
                    
                    num_projections = struct.unpack('<Q', f.read(8))[0]
                    f.read(num_projections * 8)  # 跳过投影数据
                    
                    ply_file.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")

# ================================
# 参数类和主导出函数
# ================================

class ExportSceneParams:
    """导出参数配置类"""
    
    def __init__(self):
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
        self.batch_size = 1000000

    def log(self):
        """打印参数配置"""
        print("=== Export Parameters ===")
        print(f"All chunks: {self.all_chunks}")
        print(f"All frames: {self.all_frames}")
        print(f"Zero cx/cy: {self.zero_cxy}")
        print(f"Use localframe: {self.use_localframe}")
        print(f"Image quality: {self.image_quality}")
        print(f"Export images: {self.export_images}")
        print(f"Export masks: {self.export_masks}")
        print(f"Batch size: {self.batch_size}")

def export_for_gaussian_splatting_optimized(params=ExportSceneParams()):
    """内存优化的主导出函数"""
    params.log()

    # 选择输出文件夹
    folder = Metashape.app.getExistingDirectory("Select output folder")
    if not folder:
        return False
    folder = folder + "/"

    # 获取chunk目录
    chunk_dirs = get_chunk_dirs(folder, params)
    if not chunk_dirs:
        return False

    chunk_num = len(chunk_dirs)
    
    for chunk_id, (chunk_key, chunk_folder) in enumerate(chunk_dirs.items()):
        if progress_controller.is_cancelled():
            return False
            
        # 获取chunk
        chunk = None
        for ck in Metashape.app.document.chunks:
            if ck.key == chunk_key:
                chunk = ck
                break
        if not chunk:
            continue

        frame_num = len(chunk.frames) if params.all_frames else 1
        
        for frame_id, frame in enumerate(chunk.frames):
            if progress_controller.is_cancelled():
                return False
                
            if not frame.tie_points:
                continue
            if not params.all_frames and not (frame == chunk.frame):
                continue

            folder_path = chunk_folder + ("" if frame_num == 1 else f"frame_{frame_id:06d}/")
            print(f"\nProcessing: {folder_path}")

            # 创建目录结构
            if not build_dir_structure(folder_path, params.confirm_deletion):
                return False

            # 计算校准
            progress_controller.update_progress(0, 100, f"处理 Chunk {chunk_id+1}/{chunk_num}")
            calibs = compute_undistorted_calibs(frame, params.zero_cxy)
            if progress_controller.is_cancelled():
                return False

            # 导出图像和掩膜
            save_undistorted_images(params, frame, folder_path, calibs)
            if progress_controller.is_cancelled():
                return False
                
            save_undistorted_masks(params, frame, folder_path, calibs)
            if progress_controller.is_cancelled():
                return False
            
            # 保存相机参数
            progress_controller.update_progress(50, 100, "保存相机参数...")
            save_cameras(params, folder_path, calibs)
            save_camera_params(params, folder_path, calibs)

            # 流式处理
            print("Starting memory-optimized streaming export...")
            
            # 创建相机索引
            camera_info = create_camera_projection_index(frame, calibs)
            if progress_controller.is_cancelled():
                return False
            
            # 流式保存相机位姿
            save_images_streaming(params, frame, folder_path, calibs, camera_info)
            if progress_controller.is_cancelled():
                return False
            
            # 流式保存点云
            points_saved = save_points_streaming(params, frame, folder_path, calibs, camera_info, params.batch_size)
            if progress_controller.is_cancelled():
                return False

            print(f"Export completed for frame {frame_id}: {points_saved} points")

    progress_controller.update_progress(100, 100, "导出完成！")
    return True

# ================================
# GUI界面类
# ================================

class CollapsibleGroupBox(QtWidgets.QGroupBox):
    """可折叠的组框"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
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


class ProgressDialog(QtWidgets.QDialog):
    """进度对话框，支持取消"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("导出进度")
        self.setModal(True)
        self.setMinimumSize(500, 150)
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowCloseButtonHint)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QtWidgets.QVBoxLayout()
        
        # 步骤标签
        self.step_label = QtWidgets.QLabel("准备开始...")
        self.step_label.setAlignment(QtCore.Qt.AlignCenter)
        self.step_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(self.step_label)
        
        # 进度条
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        layout.addWidget(self.progress_bar)
        
        # 详细信息
        self.detail_label = QtWidgets.QLabel("等待开始...")
        self.detail_label.setAlignment(QtCore.Qt.AlignCenter)
        self.detail_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(self.detail_label)
        
        # 时间信息
        self.time_label = QtWidgets.QLabel("")
        self.time_label.setAlignment(QtCore.Qt.AlignCenter)
        self.time_label.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(self.time_label)
        
        # 取消按钮
        button_layout = QtWidgets.QHBoxLayout()
        self.cancel_btn = QtWidgets.QPushButton("取消")
        self.cancel_btn.setFixedSize(100, 30)
        self.cancel_btn.clicked.connect(self.cancel_export)
        button_layout.addStretch()
        button_layout.addWidget(self.cancel_btn)
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
        # 设置回调
        progress_controller.set_callback(self.update_progress)
        self.start_time = time.time()
        
    def update_progress(self, value, max_value, step_name):
        """更新进度显示"""
        self.progress_bar.setMaximum(max_value)
        self.progress_bar.setValue(value)
        
        if step_name:
            self.step_label.setText(step_name)
            
        # 计算百分比
        if max_value > 0:
            percentage = (value / max_value) * 100
            self.detail_label.setText(f"进度: {value}/{max_value} ({percentage:.1f}%)")
        
        # 时间信息
        elapsed_time = time.time() - self.start_time
        if value > 0 and max_value > 0:
            estimated_total = elapsed_time * max_value / value
            remaining_time = estimated_total - elapsed_time
            self.time_label.setText(f"已用: {elapsed_time:.0f}s | 预计剩余: {remaining_time:.0f}s")
        else:
            self.time_label.setText(f"已用时: {elapsed_time:.0f}s")
            
        self.repaint()
        
    def cancel_export(self):
        """取消导出"""
        reply = QtWidgets.QMessageBox.question(
            self, "确认取消", 
            "确定要取消导出吗？\n已处理的数据可能会丢失。",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )
        
        if reply == QtWidgets.QMessageBox.Yes:
            progress_controller.cancel()
            self.step_label.setText("正在取消...")
            self.cancel_btn.setEnabled(False)

    def closeEvent(self, event):
        """阻止直接关闭"""
        event.ignore()
        self.cancel_export()


class ExportSceneGUI(QtWidgets.QDialog):
    """主GUI界面"""
    
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Export COLMAP - Memory Optimized")
        self.setMinimumSize(450, 500)
        self.setup_ui()
        
    def setup_ui(self):
        defaults = ExportSceneParams()
        layout = QtWidgets.QVBoxLayout()
        
        # 标题
        title_label = QtWidgets.QLabel("COLMAP导出 - 内存优化版")
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px; color: #2E86AB;")
        layout.addWidget(title_label)
        
        # 说明
        info_label = QtWidgets.QLabel("支持大型项目、实时进度显示、可随时取消")
        info_label.setAlignment(QtCore.Qt.AlignCenter)
        info_label.setStyleSheet("color: #666; margin-bottom: 15px;")
        layout.addWidget(info_label)

        # 基本设置
        self.setup_basic_controls(defaults)
        general_group = QtWidgets.QGroupBox("基本设置")
        general_layout = QtWidgets.QGridLayout()
        general_layout.addWidget(QtWidgets.QLabel("Chunks:"), 0, 0)
        general_layout.addWidget(self.radioBtn_allC, 0, 1)
        general_layout.addWidget(self.radioBtn_selC, 0, 2)
        general_layout.addWidget(QtWidgets.QLabel("Frames:"), 1, 0)
        general_layout.addWidget(self.radioBtn_allF, 1, 1)
        general_layout.addWidget(self.radioBtn_selF, 1, 2)
        general_layout.addWidget(QtWidgets.QLabel("Zero cx/cy:"), 2, 0)
        general_layout.addWidget(self.zcxyBox, 2, 1)
        general_layout.addWidget(QtWidgets.QLabel("Local frame:"), 3, 0)
        general_layout.addWidget(self.locFrameBox, 3, 1)
        general_layout.addWidget(QtWidgets.QLabel("Image quality:"), 4, 0)
        general_layout.addWidget(self.imgQualSpBox, 4, 1, 1, 2)
        general_group.setLayout(general_layout)

        # 高级设置
        self.setup_advanced_controls(defaults)
        self.advanced_group = CollapsibleGroupBox()
        self.advanced_group.setTitle("高级设置")
        self.advanced_group.setCheckable(True)
        self.advanced_group.setChecked(False)
        advanced_layout = QtWidgets.QGridLayout()
        advanced_layout.addWidget(QtWidgets.QLabel("Export images:"), 0, 0)
        advanced_layout.addWidget(self.expImagesBox, 0, 1)
        advanced_layout.addWidget(QtWidgets.QLabel("Export masks:"), 1, 0)
        advanced_layout.addWidget(self.expMasksBox, 1, 1)
        advanced_layout.addWidget(QtWidgets.QLabel("Batch size:"), 2, 0)
        advanced_layout.addWidget(self.batchSizeSpBox, 2, 1)
        self.advanced_group.setLayout(advanced_layout)

        # 信息框
        memory_info = QtWidgets.QLabel()
        memory_info.setText("💡 内存优化特性:\n• 分批处理大型点云\n• 实时进度显示\n• 支持随时取消\n• 生成COLMAP + PLY格式")
        memory_info.setStyleSheet("""
            background-color: #e8f4fd; border: 1px solid #bee5eb; border-radius: 5px;
            padding: 10px; margin: 5px 0px; font-size: 11px; color: #0c5460;
        """)

        # 按钮
        button_layout = QtWidgets.QHBoxLayout()
        self.btnExport = QtWidgets.QPushButton("开始导出")
        self.btnExport.setFixedSize(100, 30)
        self.btnExport.setStyleSheet("background-color: #28a745; color: white; border: none; border-radius: 5px; font-weight: bold;")
        self.btnQuit = QtWidgets.QPushButton("退出")
        self.btnQuit.setFixedSize(100, 30)
        self.btnQuit.setStyleSheet("background-color: #6c757d; color: white; border: none; border-radius: 5px;")
        
        button_layout.addWidget(self.btnExport)
        button_layout.addStretch()
        button_layout.addWidget(self.btnQuit)

        # 组装布局
        layout.addWidget(general_group)
        layout.addWidget(self.advanced_group)
        layout.addWidget(memory_info)
        layout.addStretch()
        layout.addLayout(button_layout)
        self.setLayout(layout)

        # 连接信号
        self.btnExport.clicked.connect(self.run_export)
        self.btnQuit.clicked.connect(self.close)
        
    def setup_basic_controls(self, defaults):
        """设置基本控件"""
        # Chunks选择
        self.chunk_group = QtWidgets.QButtonGroup()
        self.radioBtn_allC = QtWidgets.QRadioButton("all chunks")
        self.radioBtn_selC = QtWidgets.QRadioButton("selected")
        self.chunk_group.addButton(self.radioBtn_selC)
        self.chunk_group.addButton(self.radioBtn_allC)
        self.radioBtn_allC.setChecked(defaults.all_chunks)
        self.radioBtn_selC.setChecked(not defaults.all_chunks)

        # Frames选择
        self.frames_group = QtWidgets.QButtonGroup()
        self.radioBtn_allF = QtWidgets.QRadioButton("all frames")
        self.radioBtn_selF = QtWidgets.QRadioButton("active")
        self.frames_group.addButton(self.radioBtn_selF)
        self.frames_group.addButton(self.radioBtn_allF)
        self.radioBtn_allF.setChecked(defaults.all_frames)
        self.radioBtn_selF.setChecked(not defaults.all_frames)

        # 其他设置
        self.zcxyBox = QtWidgets.QCheckBox()
        self.zcxyBox.setChecked(defaults.zero_cxy)
        self.zcxyBox.setToolTip('Zero principal point - 适用于Gaussian Splatting')

        self.locFrameBox = QtWidgets.QCheckBox()
        self.locFrameBox.setChecked(defaults.use_localframe)
        self.locFrameBox.setToolTip('使用本地坐标系 - 解决大坐标问题')

        self.imgQualSpBox = QtWidgets.QSpinBox()
        self.imgQualSpBox.setMinimum(0)
        self.imgQualSpBox.setMaximum(100)
        self.imgQualSpBox.setValue(defaults.image_quality)
        self.imgQualSpBox.setToolTip('JPEG图像质量 (0-100)')
        
    def setup_advanced_controls(self, defaults):
        """设置高级控件"""
        self.expImagesBox = QtWidgets.QCheckBox()
        self.expImagesBox.setChecked(defaults.export_images)
        self.expImagesBox.setToolTip('导出去畸变图像')

        self.expMasksBox = QtWidgets.QCheckBox()
        self.expMasksBox.setChecked(defaults.export_masks)
        self.expMasksBox.setToolTip('导出去畸变掩膜')

        self.batchSizeSpBox = QtWidgets.QSpinBox()
        self.batchSizeSpBox.setMinimum(100000)
        self.batchSizeSpBox.setMaximum(10000000)
        self.batchSizeSpBox.setSingleStep(100000)
        self.batchSizeSpBox.setValue(defaults.batch_size)
        self.batchSizeSpBox.setToolTip('批处理大小\n较大值: 更快但占用更多内存\n较小值: 更慢但占用更少内存')

    def run_export(self):
        """运行导出"""
        # 获取参数
        params = ExportSceneParams()
        params.all_chunks = self.radioBtn_allC.isChecked()
        params.all_frames = self.radioBtn_allF.isChecked()
        params.zero_cxy = self.zcxyBox.isChecked()
        params.use_localframe = self.locFrameBox.isChecked()
        params.image_quality = self.imgQualSpBox.value()
        params.export_images = self.expImagesBox.isChecked()
        params.export_masks = self.expMasksBox.isChecked()
        params.batch_size = self.batchSizeSpBox.value()
        
        # 重置进度控制器
        global progress_controller
        progress_controller = ProgressController()
        
        # 显示进度对话框
        progress_dialog = ProgressDialog(self)
        start_time = time.time()
        
        try:
            progress_dialog.show()
            QtWidgets.QApplication.processEvents()
            
            # 执行导出
            success = export_for_gaussian_splatting_optimized(params)
            
            elapsed_time = time.time() - start_time
            progress_dialog.close()
            
            # 显示结果
            if progress_controller.is_cancelled():
                QtWidgets.QMessageBox.information(self, "导出已取消", "导出操作已被用户取消。")
            elif success:
                QtWidgets.QMessageBox.information(
                    self, "导出完成", 
                    f"导出成功完成！\n"
                    f"耗时: {elapsed_time:.1f} 秒\n\n"
                    f"输出格式:\n"
                    f"• COLMAP二进制格式\n"
                    f"• PLY点云文件\n"
                    f"• 相机参数文件"
                )
            else:
                QtWidgets.QMessageBox.warning(self, "导出失败", "导出过程中遇到错误，请检查控制台日志。")
                
        except Exception as e:
            progress_dialog.close()
            QtWidgets.QMessageBox.critical(self, "导出错误", f"导出过程中发生异常:\n{str(e)}")
        finally:
            self.close()

# ================================
# 主函数和菜单注册
# ================================

def export_for_gaussian_splatting_gui():
    """启动GUI界面"""
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
        dialog = ExportSceneGUI(parent)
        dialog.exec()
    except Exception as e:
        QtWidgets.QMessageBox.critical(None, "错误", f"启动界面时发生错误:\n{str(e)}")

# 注册菜单项
label = "Scripts/Export COLMAP (Memory Optimized)"
Metashape.app.addMenuItem(label, export_for_gaussian_splatting_gui)

print("=" * 60)
print("🚀 Memory Optimized COLMAP Export Script Loaded!")
print("=" * 60)
print("✨ Features:")
print("  • Memory-optimized batch processing")
print("  • Real-time progress display")
print("  • Cancellable at any time")
print("  • COLMAP + PLY output formats")
print("  • Support for large-scale projects")
print("")
print(f"📋 Menu: {label}")
print("🎯 Ready to export your Metashape projects!")
print("=" * 60)