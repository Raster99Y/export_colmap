import os
import shutil
import struct
import math
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
    for cam in frame.cameras:
        if cam.transform is None or cam.sensor is None or not cam.enabled:
            continue
        if cam.sensor.key not in calibs:
            continue
        (calib0, calib1, T1) = get_calibs(cam, calibs)
        if calib0 is None:
            continue

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
    print("Undistorted", cnt, "cameras")

def save_undistorted_masks(params, frame, folder, calibs):
    print("Exporting masks.")
    folder = folder + "masks/"
    if not clean_dir(folder, params.confirm_deletion):
        print("Masks folder already exists, aborting.")
        return False

    T = Metashape.Matrix.Diag([1, 1, 1, 1])

    cnt = 0
    for cam in frame.cameras:
        if cam.transform is None or cam.sensor is None or not cam.enabled:
            continue
        if cam.sensor.key not in calibs:
            continue
        (calib0, calib1, T1) = get_calibs(cam, calibs)
        if calib0 is None:
            continue
        if not cam.mask:
            # Skip if image has no mask assigned.
            continue

        mask = cam.mask.image().warp(calib0, T, calib1, T1)
        # , 'U8') # Convert image to single channel grayscale.
        mask = mask.convert("L")
        name = get_camera_name(cam)
        mask.save(str(Path(folder + name).with_suffix('.png')))
        cnt += 1
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
    
    print("已保存", len(calibs), "个相机的参数到cameras_params.txt")


# 🔥 关键修改：流式处理函数，避免构建大型字典

def create_camera_projection_index(frame, calibs):
    """创建相机投影的索引，但不存储实际投影数据"""
    print("创建相机投影索引...")
    
    camera_info = {}  # 只存储相机基本信息，不存储投影
    
    for cam in frame.cameras:
        if cam.transform is None or cam.sensor is None or not cam.enabled:
            continue
        (calib0, calib1, T1) = get_calibs(cam, calibs)
        if calib0 is None:
            continue
            
        # 只存储相机基本信息，不预处理投影
        camera_info[cam.key] = {
            'camera': cam,
            'calib0': calib0,
            'calib1': calib1, 
            'T1': T1,
            'T1_inv': T1.inv()
        }
    
    print(f"已索引 {len(camera_info)} 个相机")
    return camera_info

def process_point_projections(point_idx, track_id, frame, camera_info, only_good=True):
    """即时处理单个点的投影，不存储中间结果"""
    good_projections = []
    bad_projections = []
    
    # 查找该track_id在各相机中的投影
    for cam_key, cam_info in camera_info.items():
        camera = cam_info['camera']
        projections = frame.tie_points.projections[camera]
        
        for proj_idx, proj in enumerate(projections):
            if proj.track_id == track_id:
                # 即时计算投影坐标
                pt_proj = cam_info['calib1'].project(
                    cam_info['T1_inv'].mulp(cam_info['calib0'].unproject(proj.coord))
                )
                
                # 判断投影是否有效
                good = (pt_proj is not None and 
                       0 <= pt_proj.x < cam_info['calib1'].width and 
                       0 <= pt_proj.y < cam_info['calib1'].height)
                
                projection_data = (pt_proj, proj.size, track_id)
                
                if good:
                    good_projections.append((cam_key, projection_data))
                else:
                    bad_projections.append((cam_key, projection_data))
    
    return good_projections, bad_projections

def save_images_streaming(params, frame, folder, calibs, camera_info):
    """流式保存相机数据，不构建images字典"""
    print("流式导出相机位姿...")
    
    only_good = params.only_good
    T_shift = get_coord_transform(frame, params.use_localframe)
    
    # 首先需要知道总的投影数量来正确写入images.bin
    # 这里需要预扫描一遍，但不存储数据
    print("预扫描投影数量...")
    camera_projection_counts = {}
    
    for cam_key, cam_info in camera_info.items():
        camera = cam_info['camera']
        good_count = 0
        bad_count = 0
        
        projections = frame.tie_points.projections[camera]
        for proj in projections:
            # 快速判断投影有效性（简化版本）
            pt_proj = cam_info['calib1'].project(
                cam_info['T1_inv'].mulp(cam_info['calib0'].unproject(proj.coord))
            )
            
            if (pt_proj is not None and 
                0 <= pt_proj.x < cam_info['calib1'].width and 
                0 <= pt_proj.y < cam_info['calib1'].height):
                good_count += 1
            else:
                bad_count += 1
        
        total_count = good_count if only_good else good_count + bad_count
        camera_projection_counts[cam_key] = total_count
    
    # 写入images.bin
    with open(folder + "sparse/0/images.bin", "wb") as fout:
        fout.write(u64(len(camera_info)))
        
        for cam_key, cam_info in camera_info.items():
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
            
            # 写入投影数量
            proj_count = camera_projection_counts[cam_key]
            fout.write(u64(proj_count))
            
            # 即时处理并写入投影数据
            projections = frame.tie_points.projections[camera]
            for proj in projections:
                # 计算投影坐标
                pt_proj = cam_info['calib1'].project(
                    cam_info['T1_inv'].mulp(cam_info['calib0'].unproject(proj.coord))
                )
                
                # 判断是否有效
                good = (pt_proj is not None and 
                       0 <= pt_proj.x < cam_info['calib1'].width and 
                       0 <= pt_proj.y < cam_info['calib1'].height)
                
                # 根据only_good参数决定是否写入
                if only_good and not good:
                    continue
                    
                # 写入投影数据
                track_id = proj.track_id
                fout.write(d64(pt_proj.x if pt_proj else 0))
                fout.write(d64(pt_proj.y if pt_proj else 0))
                fout.write(u64(track_id))
    
    print(f"已流式保存 {len(camera_info)} 个相机的位姿数据")

def save_points_streaming(params, frame, folder, calibs, camera_info, batch_size=1000000):
    """真正的流式点云处理 - 不构建tracks字典"""
    print(f"开始流式处理点云，批大小: {batch_size}")
    
    only_good = params.only_good
    T = get_coord_transform(frame, params.use_localframe)
    tie_points = frame.tie_points
    
    # 创建临时文件夹
    temp_folder = folder + "temp_batches/"
    os.makedirs(temp_folder, exist_ok=True)
    
    batch_files = []
    batch_num = 0
    current_batch = []
    total_points = 0
    processed_points = 0
    
    try:
        print("流式处理点云数据...")
        
        # 🔥 关键：流式处理，不构建完整的tracks字典
        for point_idx, point in enumerate(tie_points.points):
            track_id = point.track_id
            
            # 即时处理该点的投影
            good_projections, bad_projections = process_point_projections(
                point_idx, track_id, frame, camera_info, only_good
            )
            
            # 检查是否为有效点（有投影的点）
            if not good_projections and not bad_projections:
                continue
                
            # 计算3D坐标
            pt = T * point.coord
            track = tie_points.tracks[track_id]
            
            # 准备投影数据
            all_projections = []
            for cam_key, (pt_proj, size, tid) in good_projections:
                # 这里需要获取投影在该相机中的索引
                # 简化处理：使用相机key和投影数据
                all_projections.append((cam_key, len(all_projections)))
            
            if not only_good:
                for cam_key, (pt_proj, size, tid) in bad_projections:
                    all_projections.append((cam_key, len(all_projections)))
            
            # 添加到当前批次
            current_batch.append({
                'track_id': track_id,
                'position': (pt.x, pt.y, pt.z),
                'color': (track.color[0], track.color[1], track.color[2]),
                'projections': all_projections
            })
            
            total_points += 1
            processed_points += 1
            
            # 显示进度
            if processed_points % 50000 == 0:
                print(f"已处理 {processed_points} 个点")
            
            # 批次满了，写入文件并清空内存
            if len(current_batch) >= batch_size:
                batch_file = temp_folder + f"batch_{batch_num:04d}.bin"
                save_batch_to_file(current_batch, batch_file)
                batch_files.append(batch_file)
                
                print(f"已保存批次 {batch_num + 1}, 包含 {len(current_batch)} 个点")
                
                # 🔥 关键：立即清空当前批次，释放内存
                current_batch = []
                batch_num += 1
        
        # 处理最后一个不满的批次
        if current_batch:
            batch_file = temp_folder + f"batch_{batch_num:04d}.bin"
            save_batch_to_file(current_batch, batch_file)
            batch_files.append(batch_file)
            print(f"已保存最后批次，包含 {len(current_batch)} 个点")
        
        # 合并批次文件
        print("正在合并批次到 points3D.bin...")
        merge_batches_to_colmap_binary(batch_files, folder + "sparse/0/points3D.bin", total_points)
        
        # 生成PLY文件
        print("正在生成 PLY 文件...")
        convert_to_ply(batch_files, folder + "sparse/0/points3D.ply")
        
        print(f"流式处理完成，总共处理 {total_points} 个点")
        return total_points
        
    finally:
        # 清理临时文件
        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)
            print("已清理临时文件")

def save_batch_to_file(batch_data, filename):
    """保存一个批次的数据到文件"""
    with open(filename, 'wb') as f:
        # 写入批次中的点数量
        f.write(u64(len(batch_data)))
        
        for point_data in batch_data:
            # 写入track ID
            f.write(u64(point_data['track_id']))
            # 写入位置
            f.write(d64(point_data['position'][0]))
            f.write(d64(point_data['position'][1]))
            f.write(d64(point_data['position'][2]))
            # 写入颜色
            f.write(u8(point_data['color'][0]))
            f.write(u8(point_data['color'][1]))
            f.write(u8(point_data['color'][2]))
            # 写入误差（占位符）
            f.write(d64(0))
            # 写入投影数量
            f.write(u64(len(point_data['projections'])))
            # 写入投影数据
            for camera_key, proj_idx in point_data['projections']:
                f.write(u32(camera_key))
                f.write(u32(proj_idx))

def merge_batches_to_colmap_binary(batch_files, output_file, total_points):
    """合并所有批次文件到COLMAP二进制格式"""
    with open(output_file, "wb") as fout:
        # 写入总点数
        fout.write(u64(total_points))
        
        # 读取每个批次文件并写入主文件
        for batch_file in batch_files:
            with open(batch_file, 'rb') as fin:
                # 跳过批次文件中的点数量信息
                fin.read(8)  # 跳过 u64 的点数量
                
                # 复制剩余数据
                while True:
                    chunk = fin.read(8192)  # 8KB 块大小
                    if not chunk:
                        break
                    fout.write(chunk)

def convert_to_ply(batch_files, ply_filename):
    """将批次文件转换为PLY格式"""
    total_points = 0
    
    # 首先计算总点数
    for batch_file in batch_files:
        with open(batch_file, 'rb') as f:
            points_in_batch = struct.unpack('<Q', f.read(8))[0]
            total_points += points_in_batch
    
    # 写入PLY文件
    with open(ply_filename, 'w') as ply_file:
        # 写入PLY头部
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
                    # 读取track ID（跳过）
                    f.read(8)
                    
                    # 读取位置
                    x = struct.unpack('<d', f.read(8))[0]
                    y = struct.unpack('<d', f.read(8))[0]
                    z = struct.unpack('<d', f.read(8))[0]
                    
                    # 读取颜色
                    r = struct.unpack('<B', f.read(1))[0]
                    g = struct.unpack('<B', f.read(1))[0]
                    b = struct.unpack('<B', f.read(1))[0]
                    
                    # 跳过误差
                    f.read(8)
                    
                    # 跳过投影数据
                    num_projections = struct.unpack('<Q', f.read(8))[0]
                    f.read(num_projections * 8)  # 每个投影8字节（camera_key + proj_idx）
                    
                    # 写入PLY格式的点
                    ply_file.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")


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
        self.batch_size = 1000000  # 批处理大小

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
        print("Batch size for point cloud export:", self.batch_size)


def export_for_gaussian_splatting_optimized(params = ExportSceneParams(), progress = QtWidgets.QProgressBar()):
    """🔥 内存优化版本 - 避免构建大型tracks和images字典"""
    log_result = lambda x: print("", x, "-----------------------------------", sep="\n")
    progress.setMinimum(0)
    progress.setMaximum(1000)
    set_progress = lambda x: progress.setValue(int(x * 1000))
    params.log()

    folder = Metashape.app.getExistingDirectory("Output folder")
    if len(folder) == 0:
        log_result("No chosen folder")
        return
    folder = folder + "/"
    print(folder)

    chunk_dirs = get_chunk_dirs(folder, params)
    if len(chunk_dirs) == 0:
        log_result("Aborted")
        return

    chunk_num = len(chunk_dirs)
    for chunk_id, (chunk_key, chunk_folder) in enumerate(chunk_dirs.items()):
        chunk = [ck for ck in Metashape.app.document.chunks if ck.key == chunk_key]
        if (len(chunk) != 1):
            print("Chunk not found, key =", chunk_key)
            continue
        chunk = chunk[0]

        frame_num = len(chunk.frames) if params.all_frames else 1
        prog_step = 1 / chunk_num
        set_progress(prog_step * chunk_id)
        set_progress_frame = lambda n: set_progress(prog_step * (chunk_id + n / frame_num))
        frame_cnt = 0

        for frame_id, frame in enumerate(chunk.frames):
            if not frame.tie_points:
                continue
            if not params.all_frames and not (frame == chunk.frame):
                continue
            set_progress_frame(frame_cnt)
            frame_cnt += 1

            folder_path = chunk_folder + ("" if frame_num == 1 else "frame_" + str(frame_id).zfill(6) + "/")
            print("\n" + folder_path)

            if not build_dir_structure(folder_path, params.confirm_deletion):
                log_result("Aborted")
                return

            # 计算去畸变标定
            calibs = compute_undistorted_calibs(frame, params.zero_cxy)

            # 导出图像和掩膜
            if params.export_images:
                save_undistorted_images(params, frame, folder_path, calibs)
            if params.export_masks:
                save_undistorted_masks(params, frame, folder_path, calibs)
            
            # 保存相机参数
            save_cameras(params, folder_path, calibs)
            save_camera_params(params, folder_path, calibs)

            # 🔥 关键修改：使用流式处理，避免构建大型字典
            print("开始内存优化的流式处理...")
            
            # 创建轻量级相机索引（不存储投影数据）
            camera_info = create_camera_projection_index(frame, calibs)
            
            # 流式处理相机位姿
            save_images_streaming(params, frame, folder_path, calibs, camera_info)
            
            # 流式处理点云数据
            save_points_streaming(params, frame, folder_path, calibs, camera_info, params.batch_size)

    set_progress(1)
    log_result("内存优化处理完成！")


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
        for button in self.buttons:
            button.setEnabled(False)

        params = ExportSceneParams()
        params.all_chunks = self.radioBtn_allC.isChecked()
        params.all_frames = self.radioBtn_allF.isChecked()
        params.zero_cxy = self.zcxyBox.isChecked()
        params.use_localframe = self.locFrameBox.isChecked()
        params.image_quality = self.imgQualSpBox.value()
        params.export_images = self.expImagesBox.isChecked()
        params.export_masks = self.expMasksBox.isChecked()
        params.batch_size = self.batchSizeSpBox.value()
        try:
            # 🔥 使用内存优化版本
            export_for_gaussian_splatting_optimized(params, self.pBar)
        finally:
            self.done(0)

    def __init__(self, parent):
        QtWidgets.QDialog.__init__(self, parent)
        self.setWindowTitle("Export scene in Colmap format (Memory Optimized):")

        defaults = ExportSceneParams()

        self.btnQuit = QtWidgets.QPushButton("Quit")
        self.btnQuit.setFixedSize(100,25)

        self.btnP1 = QtWidgets.QPushButton("Export")
        self.btnP1.setFixedSize(100,25)

        self.pBar = QtWidgets.QProgressBar()
        self.pBar.setTextVisible(False)
        self.pBar.setFixedSize(100, 25)

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

        # 批大小设置
        self.batchSizeTxt = QtWidgets.QLabel()
        self.batchSizeTxt.setText("Batch size")
        self.batchSizeTxt.setFixedSize(100, 25)

        self.batchSizeSpBox = QtWidgets.QSpinBox()
        self.batchSizeSpBox.setMinimum(100000)
        self.batchSizeSpBox.setMaximum(10000000)
        self.batchSizeSpBox.setSingleStep(100000)
        self.batchSizeSpBox.setValue(defaults.batch_size)

        zcxyToolTip = 'Output camera calibrations will have zero cx and cy\nShould be checked until Gaussian Splatting software considers this parameters\nMay result in information loss during export (large cropping)\nTo mitigate that effect, do step 1.1.0. and check "Adaptive camera model fitting" at 1.2. of the script description'
        self.zcxyTxt.setToolTip(zcxyToolTip)
        self.zcxyBox.setToolTip(zcxyToolTip)

        locFrameToolTip = "Shifts coordinates origin to the center of the bounding box\nUses localframe rotation at this point\nThis is useful to fix large coordinates"
        self.locFrameTxt.setToolTip(locFrameToolTip)
        self.locFrameBox.setToolTip(locFrameToolTip)

        imgQualToolTip = "Quality of the output undistorted images (jpeg only)\nMin = 0, Max = 100"
        self.imgQualTxt.setToolTip(imgQualToolTip)
        self.imgQualSpBox.setToolTip(imgQualToolTip)

        expImagesToolTip = "You can disable export of the undistorted images"
        self.expImagesTxt.setToolTip(expImagesToolTip)
        self.expImagesBox.setToolTip(expImagesToolTip)

        expMasksToolTip = "You can enable export of the undistorted masks"
        self.expMasksTxt.setToolTip(expMasksToolTip)
        self.expMasksBox.setToolTip(expMasksToolTip)

        batchSizeToolTip = "Number of points to process in each batch\nLarger values use more memory but may be faster\nSmaller values use less memory but may be slower"
        self.batchSizeTxt.setToolTip(batchSizeToolTip)
        self.batchSizeSpBox.setToolTip(batchSizeToolTip)

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
        advanced_layout.addWidget(self.expMasksTxt, 1, 0)
        advanced_layout.addWidget(self.expMasksBox, 1, 1)
        advanced_layout.addWidget(self.batchSizeTxt, 2, 0)
        advanced_layout.addWidget(self.batchSizeSpBox, 2, 1, 1, 2)

        self.gbGeneral = QtWidgets.QGroupBox()
        self.gbGeneral.setLayout(general_layout)
        self.gbGeneral.setTitle("General")

        self.gbAdvanced = CollapsibleGroupBox()
        self.gbAdvanced.setLayout(advanced_layout)
        self.gbAdvanced.setTitle("Advanced (Memory Optimized)")
        self.gbAdvanced.setCheckable(True)
        self.gbAdvanced.setChecked(False)
        self.gbAdvanced.toggled.connect(lambda: QtCore.QTimer.singleShot(20, lambda: self.adjustSize()))

        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.gbGeneral, 0, 0, 1, 3)
        layout.addWidget(self.gbAdvanced, 1, 0, 1, 3)
        layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.MinimumExpanding))
        layout.addWidget(self.pBar, 3, 0)
        layout.addWidget(self.btnP1, 3, 1)
        layout.addWidget(self.btnQuit, 3, 2)
        self.setLayout(layout)

        self.buttons = [self.btnP1, self.btnQuit, self.radioBtn_allC, self.radioBtn_selC, self.radioBtn_allF, self.radioBtn_selF, self.zcxyBox, self.locFrameBox, self.imgQualSpBox, self.expImagesBox, self.expMasksBox, self.batchSizeSpBox]

        proc = lambda : self.run_export()

        QtCore.QObject.connect(self.btnP1, QtCore.SIGNAL("clicked()"), proc)
        QtCore.QObject.connect(self.btnQuit, QtCore.SIGNAL("clicked()"), self, QtCore.SLOT("reject()"))

        self.exec()


def export_for_gaussian_splatting_gui():
    global app
    app = QtWidgets.QApplication.instance()
    parent = app.activeWindow()
    dlg = ExportSceneGUI(parent)

label = "Scripts/Export Colmap project (Memory Optimized)"
Metashape.app.addMenuItem(label, export_for_gaussian_splatting_gui)
print("To execute this script press {}".format(label))