import os
import sys
import psutil
import gc
from PySide2 import QtCore, QtGui, QtWidgets


# Checking compatibility
compatible_major_version = "2.1"
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))


def get_memory_usage():
    """获取当前进程的内存使用情况"""
    process = psutil.Process()
    memory_info = process.memory_info()
    return {
        'rss': memory_info.rss,  # 物理内存 (字节)
        'vms': memory_info.vms,  # 虚拟内存 (字节)
        'percent': process.memory_percent(),  # 内存使用百分比
        'available': psutil.virtual_memory().available  # 可用内存
    }

def format_bytes(bytes_value):
    """格式化字节数为人类可读格式"""
    if bytes_value == 0:
        return "0 B"
    
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    unit_index = 0
    value = float(bytes_value)
    
    while value >= 1024 and unit_index < len(units) - 1:
        value /= 1024
        unit_index += 1
    
    return f"{value:.2f} {units[unit_index]}"

def estimate_object_memory_size(obj, max_depth=3, visited=None):
    """估算Python对象的内存大小（递归）"""
    if visited is None:
        visited = set()
    
    if max_depth <= 0 or id(obj) in visited:
        return 0
    
    visited.add(id(obj))
    size = sys.getsizeof(obj)
    
    # 根据对象类型进行递归估算
    if isinstance(obj, dict):
        size += sum(estimate_object_memory_size(k, max_depth-1, visited) + 
                   estimate_object_memory_size(v, max_depth-1, visited) 
                   for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set)):
        size += sum(estimate_object_memory_size(item, max_depth-1, visited) 
                   for item in obj)
    elif hasattr(obj, '__dict__'):
        size += estimate_object_memory_size(obj.__dict__, max_depth-1, visited)
    
    return size

def get_tie_points_memory_estimate(tie_points):
    """估算tie_points的内存使用"""
    if not tie_points:
        return 0
    
    memory_estimate = 0
    
    try:
        # 估算points数组内存
        if hasattr(tie_points, 'points') and tie_points.points:
            points_count = len(tie_points.points)
            # 每个点大约包含: 3D坐标(24字节) + track_id(8字节) + 其他属性(约32字节)
            memory_estimate += points_count * 64
        
        # 估算tracks数组内存  
        if hasattr(tie_points, 'tracks') and tie_points.tracks:
            tracks_count = len(tie_points.tracks)
            # 每个track大约包含: 颜色(3字节) + 置信度(8字节) + 其他属性(约32字节)
            memory_estimate += tracks_count * 48
        
        # 估算projections内存（这通常是最大的部分）
        if hasattr(tie_points, 'projections'):
            total_projections = 0
            try:
                # 遍历所有相机的投影
                for camera_projections in tie_points.projections.values():
                    if camera_projections:
                        total_projections += len(camera_projections)
                # 每个投影: 2D坐标(16字节) + track_id(8字节) + size(8字节) + 其他(32字节)
                memory_estimate += total_projections * 64
            except:
                # 如果无法遍历，使用估算
                memory_estimate += points_count * 10 * 64  # 假设每个点平均在10个相机中可见
    
    except Exception as e:
        print(f"估算tie_points内存时出错: {e}")
        return 0
    
    return memory_estimate

def estimate_export_colmap_memory_usage(frame, calibs=None):
    """估算按照export_colmap原版脚本导出时tracks和images字典的内存占用"""
    if not frame or not hasattr(frame, 'tie_points') or not frame.tie_points:
        return {
            'tracks_dict_memory': 0,
            'images_dict_memory': 0,
            'total_export_memory': 0,
            'memory_breakdown': {}
        }
    
    tie_points = frame.tie_points
    memory_breakdown = {}
    
    try:
        # 模拟tracks字典的内存使用
        tracks_memory = 0
        points_count = 0
        tracks_count = 0
        
        if hasattr(tie_points, 'points') and tie_points.points:
            points_count = len(tie_points.points)
            
        if hasattr(tie_points, 'tracks') and tie_points.tracks:
            tracks_count = len(tie_points.tracks)
        
        # tracks字典结构: { track_id: [ point indices, good projections, bad projections ] }
        # 每个track_id条目的内存估算:
        for track_id in range(tracks_count):
            # Python字典条目开销: key(8字节) + value指针(8字节) + 哈希表开销(约16字节)
            dict_entry_overhead = 32
            
            # 列表结构: [point_indices, good_projections, bad_projections]
            # 3个列表对象: 每个列表约48字节基础开销
            lists_overhead = 3 * 48
            
            # point_indices列表: 通常每个track只有1-3个点索引
            point_indices_memory = 2 * 8  # 平均2个点索引，每个8字节
            
            # 投影列表: 估算每个track在多少相机中可见
            avg_projections_per_track = max(1, points_count // tracks_count * 8)  # 假设平均可见度
            good_projections_memory = avg_projections_per_track * 16  # 每个投影(camera_key, proj_idx)
            bad_projections_memory = avg_projections_per_track * 0.2 * 16  # 假设20%为bad投影
            
            track_total_memory = (dict_entry_overhead + lists_overhead + 
                                 point_indices_memory + good_projections_memory + bad_projections_memory)
            tracks_memory += track_total_memory
        
        # Python字典本身的开销 (哈希表、桶数组等)
        tracks_dict_overhead = tracks_count * 1.33 * 8  # 哈希表负载因子约0.75，加上额外开销
        tracks_memory += tracks_dict_overhead
        
        memory_breakdown['tracks_dict_entries'] = tracks_memory - tracks_dict_overhead
        memory_breakdown['tracks_dict_overhead'] = tracks_dict_overhead
        
        # 模拟images字典的内存使用
        images_memory = 0
        cameras_count = 0
        
        # 获取相机数量
        if hasattr(frame, 'cameras') and frame.cameras:
            cameras_count = len([cam for cam in frame.cameras 
                               if cam.transform is not None and cam.sensor is not None and cam.enabled])
        
        # images字典结构: { camera_key: [ camera, good projections, bad projections ] }
        total_projections = 0
        if hasattr(tie_points, 'projections'):
            try:
                for camera_projections in tie_points.projections.values():
                    if camera_projections:
                        total_projections += len(camera_projections)
            except:
                total_projections = points_count * 8  # 估算值
        
        avg_projections_per_camera = total_projections // max(1, cameras_count) if cameras_count > 0 else 0
        
        for cam_id in range(cameras_count):
            # Python字典条目开销
            dict_entry_overhead = 32
            
            # 列表结构: [camera, good_projections, bad_projections] 
            lists_overhead = 48  # 一个列表对象
            camera_ref_memory = 8  # 相机对象引用
            
            # 投影数据: (undistorted_pt, size, track_id) 元组
            # 每个投影元组: Vector对象(约64字节) + float(8字节) + int(8字节) = 80字节
            good_projections_memory = avg_projections_per_camera * 0.8 * 80  # 假设80%为good
            bad_projections_memory = avg_projections_per_camera * 0.2 * 80   # 假设20%为bad
            
            # 好投影和坏投影的列表开销
            projection_lists_overhead = 2 * 48  # 两个子列表
            
            camera_total_memory = (dict_entry_overhead + lists_overhead + camera_ref_memory +
                                 good_projections_memory + bad_projections_memory + projection_lists_overhead)
            images_memory += camera_total_memory
        
        # images字典本身的开销
        images_dict_overhead = cameras_count * 1.33 * 8
        images_memory += images_dict_overhead
        
        memory_breakdown['images_dict_entries'] = images_memory - images_dict_overhead
        memory_breakdown['images_dict_overhead'] = images_dict_overhead
        
        # 额外的处理开销（临时变量、计算缓存等）
        processing_overhead = (tracks_memory + images_memory) * 0.15
        memory_breakdown['processing_overhead'] = processing_overhead
        
        total_export_memory = tracks_memory + images_memory + processing_overhead
        
        # 详细分解
        memory_breakdown.update({
            'points_count': points_count,
            'tracks_count': tracks_count,
            'cameras_count': cameras_count,
            'total_projections': total_projections,
            'avg_projections_per_camera': avg_projections_per_camera,
            'tracks_memory_per_entry': tracks_memory / max(1, tracks_count),
            'images_memory_per_camera': images_memory / max(1, cameras_count)
        })
        
        return {
            'tracks_dict_memory': tracks_memory,
            'images_dict_memory': images_memory,
            'total_export_memory': total_export_memory,
            'memory_breakdown': memory_breakdown
        }
        
    except Exception as e:
        print(f"估算export_colmap内存使用时出错: {e}")
        return {
            'tracks_dict_memory': 0,
            'images_dict_memory': 0,
            'total_export_memory': 0,
            'memory_breakdown': {'error': str(e)}
        }

def analyze_frame_statistics(frame):
    """分析单个frame的统计信息"""
    stats = {
        'points_count': 0,
        'tracks_count': 0,
        'projections_count': 0,
        'cameras_with_projections': 0,
        'estimated_memory': 0,
        'export_colmap_memory': {}  # 🆕 新增export_colmap内存估算
    }
    
    if not frame:
        return stats
    
    try:
        # 分析tie_points
        if hasattr(frame, 'tie_points') and frame.tie_points:
            tie_points = frame.tie_points
            
            # 统计points
            if hasattr(tie_points, 'points') and tie_points.points:
                stats['points_count'] = len(tie_points.points)
            
            # 统计tracks
            if hasattr(tie_points, 'tracks') and tie_points.tracks:
                stats['tracks_count'] = len(tie_points.tracks)
            
            # 统计projections
            if hasattr(tie_points, 'projections'):
                total_projections = 0
                cameras_with_proj = 0
                try:
                    for camera_projections in tie_points.projections.values():
                        if camera_projections and len(camera_projections) > 0:
                            total_projections += len(camera_projections)
                            cameras_with_proj += 1
                except:
                    pass
                
                stats['projections_count'] = total_projections
                stats['cameras_with_projections'] = cameras_with_proj
            
            # 估算原始内存使用
            stats['estimated_memory'] = get_tie_points_memory_estimate(tie_points)
            
            # 🆕 估算export_colmap脚本的内存使用
            stats['export_colmap_memory'] = estimate_export_colmap_memory_usage(frame)
    
    except Exception as e:
        print(f"分析frame时出错: {e}")
    
    return stats

def analyze_chunk_statistics(chunk):
    """分析单个chunk的统计信息"""
    stats = {
        'frames_count': 0,
        'total_points': 0,
        'total_tracks': 0,
        'total_projections': 0,
        'cameras_count': 0,
        'sensors_count': 0,
        'estimated_memory': 0,
        'frames_stats': [],
        'total_export_colmap_memory': 0,  # 🆕 总的export_colmap内存
        'export_memory_breakdown': {}     # 🆕 详细内存分解
    }
    
    if not chunk:
        return stats
    
    try:
        # 统计相机数量
        if hasattr(chunk, 'cameras') and chunk.cameras:
            stats['cameras_count'] = len(chunk.cameras)
        
        # 统计传感器数量
        if hasattr(chunk, 'sensors') and chunk.sensors:
            stats['sensors_count'] = len(chunk.sensors)
        
        # 分析frames
        if hasattr(chunk, 'frames') and chunk.frames:
            stats['frames_count'] = len(chunk.frames)
            
            total_tracks_memory = 0
            total_images_memory = 0
            total_export_memory = 0
            
            for i, frame in enumerate(chunk.frames):
                frame_stats = analyze_frame_statistics(frame)
                frame_stats['frame_index'] = i
                stats['frames_stats'].append(frame_stats)
                
                # 累计统计
                stats['total_points'] += frame_stats['points_count']
                stats['total_tracks'] += frame_stats['tracks_count']
                stats['total_projections'] += frame_stats['projections_count']
                stats['estimated_memory'] += frame_stats['estimated_memory']
                
                # 🆕 累计export_colmap内存统计
                export_mem = frame_stats['export_colmap_memory']
                if export_mem:
                    total_tracks_memory += export_mem.get('tracks_dict_memory', 0)
                    total_images_memory += export_mem.get('images_dict_memory', 0)
                    total_export_memory += export_mem.get('total_export_memory', 0)
            
            # 🆕 设置总的export内存统计
            stats['total_export_colmap_memory'] = total_export_memory
            stats['export_memory_breakdown'] = {
                'total_tracks_dict_memory': total_tracks_memory,
                'total_images_dict_memory': total_images_memory,
                'memory_amplification_ratio': total_export_memory / max(1, stats['estimated_memory']),
                'tracks_dict_percentage': (total_tracks_memory / max(1, total_export_memory)) * 100,
                'images_dict_percentage': (total_images_memory / max(1, total_export_memory)) * 100
            }
    
    except Exception as e:
        print(f"分析chunk时出错: {e}")
    
    return stats

def assess_memory_risk(peak_memory_bytes):
    """评估内存使用风险"""
    peak_memory_gb = peak_memory_bytes / (1024**3)
    
    if peak_memory_gb < 8:
        return "低风险 - 大多数系统可以处理"
    elif peak_memory_gb < 16:
        return "中等风险 - 需要16GB+内存"
    elif peak_memory_gb < 32:
        return "高风险 - 需要32GB+内存，可能出现内存交换"
    elif peak_memory_gb < 64:
        return "极高风险 - 需要64GB+内存，很可能崩溃"
    else:
        return "危险 - 几乎肯定会崩溃，需要内存优化版本"

def analyze_document_statistics():
    """分析整个document的统计信息"""
    print("开始分析Metashape项目统计信息...")
    
    doc = Metashape.app.document
    
    # 获取初始内存使用情况
    initial_memory = get_memory_usage()
    
    stats = {
        'chunks_count': 0,
        'total_frames': 0,
        'total_points': 0,
        'total_tracks': 0,
        'total_projections': 0,
        'total_cameras': 0,
        'total_sensors': 0,
        'estimated_memory': 0,
        'chunks_stats': [],
        'memory_info': initial_memory,
        'total_export_colmap_memory': 0,    # 🆕 总的export_colmap内存
        'export_memory_summary': {}         # 🆕 export内存摘要
    }
    
    try:
        # 分析chunks
        if hasattr(doc, 'chunks') and doc.chunks:
            stats['chunks_count'] = len(doc.chunks)
            
            total_export_memory = 0
            total_tracks_memory = 0
            total_images_memory = 0
            
            for i, chunk in enumerate(doc.chunks):
                print(f"正在分析chunk {i+1}/{stats['chunks_count']}...")
                
                chunk_stats = analyze_chunk_statistics(chunk)
                chunk_stats['chunk_index'] = i
                chunk_stats['chunk_label'] = chunk.label if hasattr(chunk, 'label') else f"Chunk_{i}"
                stats['chunks_stats'].append(chunk_stats)
                
                # 累计统计
                stats['total_frames'] += chunk_stats['frames_count']
                stats['total_points'] += chunk_stats['total_points']
                stats['total_tracks'] += chunk_stats['total_tracks']
                stats['total_projections'] += chunk_stats['total_projections']
                stats['total_cameras'] += chunk_stats['cameras_count']
                stats['total_sensors'] += chunk_stats['sensors_count']
                stats['estimated_memory'] += chunk_stats['estimated_memory']
                
                # 🆕 累计export_colmap内存统计
                total_export_memory += chunk_stats['total_export_colmap_memory']
                if 'export_memory_breakdown' in chunk_stats:
                    breakdown = chunk_stats['export_memory_breakdown']
                    total_tracks_memory += breakdown.get('total_tracks_dict_memory', 0)
                    total_images_memory += breakdown.get('total_images_dict_memory', 0)
            
            # 🆕 设置总的export内存摘要
            stats['total_export_colmap_memory'] = total_export_memory
            current_memory = stats['estimated_memory']
            
            stats['export_memory_summary'] = {
                'total_tracks_dict_memory': total_tracks_memory,
                'total_images_dict_memory': total_images_memory,
                'current_metashape_memory': current_memory,
                'export_script_additional_memory': total_export_memory,
                'peak_memory_during_export': current_memory + total_export_memory,
                'memory_amplification_factor': (total_export_memory / max(1, current_memory)),
                'tracks_dict_percentage': (total_tracks_memory / max(1, total_export_memory)) * 100,
                'images_dict_percentage': (total_images_memory / max(1, total_export_memory)) * 100,
                'risk_assessment': assess_memory_risk(current_memory + total_export_memory)
            }
    
    except Exception as e:
        print(f"分析document时出错: {e}")
    
    return stats

def export_statistics_to_file(stats, filepath):
    """将统计信息导出到文件"""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("Metashape项目统计报告（包含export_colmap内存分析）\n")
            f.write("=" * 60 + "\n\n")
            
            # 内存信息
            memory_info = stats['memory_info']
            f.write("系统内存信息:\n")
            f.write(f"  物理内存使用: {format_bytes(memory_info['rss'])}\n")
            f.write(f"  虚拟内存使用: {format_bytes(memory_info['vms'])}\n")
            f.write(f"  内存使用百分比: {memory_info['percent']:.2f}%\n")
            f.write(f"  可用内存: {format_bytes(memory_info['available'])}\n\n")
            
            # 总体统计
            f.write("项目总体统计:\n")
            f.write(f"  Chunks数量: {stats['chunks_count']}\n")
            f.write(f"  Frames总数: {stats['total_frames']}\n")
            f.write(f"  Points总数: {stats['total_points']:,}\n")
            f.write(f"  Tracks总数: {stats['total_tracks']:,}\n")
            f.write(f"  Projections总数: {stats['total_projections']:,}\n")
            f.write(f"  Cameras总数: {stats['total_cameras']}\n")
            f.write(f"  Sensors总数: {stats['total_sensors']}\n")
            f.write(f"  当前Metashape内存: {format_bytes(stats['estimated_memory'])}\n\n")
            
            # 🆕 Export_colmap内存分析
            if 'export_memory_summary' in stats:
                export_summary = stats['export_memory_summary']
                f.write("Export_colmap脚本内存分析:\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Tracks字典内存: {format_bytes(export_summary.get('total_tracks_dict_memory', 0))}\n")
                f.write(f"  Images字典内存: {format_bytes(export_summary.get('total_images_dict_memory', 0))}\n")
                f.write(f"  脚本额外内存: {format_bytes(export_summary.get('export_script_additional_memory', 0))}\n")
                f.write(f"  导出时峰值内存: {format_bytes(export_summary.get('peak_memory_during_export', 0))}\n")
                f.write(f"  内存放大倍数: {export_summary.get('memory_amplification_factor', 0):.2f}x\n")
                f.write(f"  Tracks字典占比: {export_summary.get('tracks_dict_percentage', 0):.1f}%\n")
                f.write(f"  Images字典占比: {export_summary.get('images_dict_percentage', 0):.1f}%\n")
                f.write(f"  风险评估: {export_summary.get('risk_assessment', 'Unknown')}\n\n")
            
            # 详细chunk统计
            for chunk_stats in stats['chunks_stats']:
                f.write(f"Chunk {chunk_stats['chunk_index']}: {chunk_stats['chunk_label']}\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Frames数量: {chunk_stats['frames_count']}\n")
                f.write(f"  Points总数: {chunk_stats['total_points']:,}\n")
                f.write(f"  Tracks总数: {chunk_stats['total_tracks']:,}\n")
                f.write(f"  Projections总数: {chunk_stats['total_projections']:,}\n")
                f.write(f"  Cameras数量: {chunk_stats['cameras_count']}\n")
                f.write(f"  Sensors数量: {chunk_stats['sensors_count']}\n")
                f.write(f"  Metashape内存: {format_bytes(chunk_stats['estimated_memory'])}\n")
                
                # 🆕 Chunk级别的export_colmap内存分析
                if 'export_memory_breakdown' in chunk_stats:
                    breakdown = chunk_stats['export_memory_breakdown']
                    f.write(f"  Export_colmap分析:\n")
                    f.write(f"    Tracks字典: {format_bytes(breakdown.get('total_tracks_dict_memory', 0))}\n")
                    f.write(f"    Images字典: {format_bytes(breakdown.get('total_images_dict_memory', 0))}\n")
                    f.write(f"    内存放大: {breakdown.get('memory_amplification_ratio', 0):.2f}x\n")
                
                # Frame详细信息
                if chunk_stats['frames_stats']:
                    f.write("  Frame详情:\n")
                    for frame_stats in chunk_stats['frames_stats']:
                        export_mem = frame_stats.get('export_colmap_memory', {})
                        f.write(f"    Frame {frame_stats['frame_index']}: ")
                        f.write(f"Points={frame_stats['points_count']:,}, ")
                        f.write(f"Tracks={frame_stats['tracks_count']:,}, ")
                        f.write(f"Projections={frame_stats['projections_count']:,}, ")
                        f.write(f"Metashape内存={format_bytes(frame_stats['estimated_memory'])}")
                        if export_mem:
                            f.write(f", Export内存={format_bytes(export_mem.get('total_export_memory', 0))}")
                        f.write("\n")
                f.write("\n")
        
        return True
    except Exception as e:
        print(f"导出统计信息时出错: {e}")
        return False

class ProjectStatsGUI(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(ProjectStatsGUI, self).__init__(parent)
        self.setWindowTitle("Metashape项目统计分析 (含Export_colmap内存分析)")
        self.setMinimumSize(900, 700)
        self.stats = None
        self.setup_ui()

    def setup_ui(self):
        layout = QtWidgets.QVBoxLayout()
        
        # 标题
        title_label = QtWidgets.QLabel("Metashape项目统计分析")
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        layout.addWidget(title_label)
        
        # 说明文本
        info_label = QtWidgets.QLabel("本工具可分析项目数据结构并预测export_colmap脚本的内存使用")
        info_label.setAlignment(QtCore.Qt.AlignCenter)
        info_label.setStyleSheet("color: #666; margin-bottom: 10px;")
        layout.addWidget(info_label)
        
        # 按钮区域
        button_layout = QtWidgets.QHBoxLayout()
        
        self.analyze_btn = QtWidgets.QPushButton("开始分析")
        self.analyze_btn.setFixedSize(120, 30)
        self.analyze_btn.clicked.connect(self.run_analysis)
        
        self.export_btn = QtWidgets.QPushButton("导出报告")
        self.export_btn.setFixedSize(120, 30)
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.export_report)
        
        self.close_btn = QtWidgets.QPushButton("关闭")
        self.close_btn.setFixedSize(120, 30)
        self.close_btn.clicked.connect(self.close)
        
        button_layout.addWidget(self.analyze_btn)
        button_layout.addWidget(self.export_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
        
        # 进度条
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # 结果显示区域
        self.result_text = QtWidgets.QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setFont(QtGui.QFont("Consolas", 9))
        layout.addWidget(self.result_text)
        
        self.setLayout(layout)

    def run_analysis(self):
        self.analyze_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # 不确定进度
        self.result_text.clear()
    def run_analysis(self):
        self.analyze_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # 不确定进度
        self.result_text.clear()
        self.result_text.append("正在分析项目...")
        
        QtWidgets.QApplication.processEvents()
        
        try:
            # 运行分析
            self.stats = analyze_document_statistics()
            self.display_results()
            self.export_btn.setEnabled(True)
            
        except Exception as e:
            self.result_text.append(f"\n分析过程中出现错误: {e}")
            
        finally:
            self.progress_bar.setVisible(False)
            self.analyze_btn.setEnabled(True)

    def display_results(self):
        if not self.stats:
            return
        
        self.result_text.clear()
        
        # 系统内存信息
        memory_info = self.stats['memory_info']
        self.result_text.append("=== 系统内存信息 ===")
        self.result_text.append(f"物理内存使用: {format_bytes(memory_info['rss'])}")
        self.result_text.append(f"虚拟内存使用: {format_bytes(memory_info['vms'])}")
        self.result_text.append(f"内存使用百分比: {memory_info['percent']:.2f}%")
        self.result_text.append(f"可用内存: {format_bytes(memory_info['available'])}")
        self.result_text.append("")
        
        # 项目总体统计
        self.result_text.append("=== 项目总体统计 ===")
        self.result_text.append(f"Chunks数量: {self.stats['chunks_count']}")
        self.result_text.append(f"Frames总数: {self.stats['total_frames']}")
        self.result_text.append(f"Points总数: {self.stats['total_points']:,}")
        self.result_text.append(f"Tracks总数: {self.stats['total_tracks']:,}")
        self.result_text.append(f"Projections总数: {self.stats['total_projections']:,}")
        self.result_text.append(f"Cameras总数: {self.stats['total_cameras']}")
        self.result_text.append(f"Sensors总数: {self.stats['total_sensors']}")
        self.result_text.append(f"当前Metashape内存: {format_bytes(self.stats['estimated_memory'])}")
        self.result_text.append("")
        
        # 🆕 Export_colmap内存分析
        if 'export_memory_summary' in self.stats:
            export_summary = self.stats['export_memory_summary']
            self.result_text.append("=== Export_colmap脚本内存分析 ===")
            self.result_text.append(f"Tracks字典预计内存: {format_bytes(export_summary.get('total_tracks_dict_memory', 0))}")
            self.result_text.append(f"Images字典预计内存: {format_bytes(export_summary.get('total_images_dict_memory', 0))}")
            self.result_text.append(f"脚本额外内存需求: {format_bytes(export_summary.get('export_script_additional_memory', 0))}")
            self.result_text.append(f"导出时峰值内存: {format_bytes(export_summary.get('peak_memory_during_export', 0))}")
            self.result_text.append(f"内存放大倍数: {export_summary.get('memory_amplification_factor', 0):.2f}x")
            
            # 风险评估用颜色标注
            risk = export_summary.get('risk_assessment', 'Unknown')
            if "低风险" in risk:
                risk_color = "🟢"
            elif "中等风险" in risk:
                risk_color = "🟡"
            elif "高风险" in risk:
                risk_color = "🟠"
            else:
                risk_color = "🔴"
            
            self.result_text.append(f"风险评估: {risk_color} {risk}")
            self.result_text.append("")
            
            # 内存使用分解
            tracks_pct = export_summary.get('tracks_dict_percentage', 0)
            images_pct = export_summary.get('images_dict_percentage', 0)
            self.result_text.append("内存使用分解:")
            self.result_text.append(f"  Tracks字典: {tracks_pct:.1f}%")
            self.result_text.append(f"  Images字典: {images_pct:.1f}%")
            self.result_text.append(f"  其他开销: {(100-tracks_pct-images_pct):.1f}%")
            self.result_text.append("")
        
        # 详细chunk统计
        self.result_text.append("=== Chunk详细统计 ===")
        for chunk_stats in self.stats['chunks_stats']:
            self.result_text.append(f"Chunk {chunk_stats['chunk_index']}: {chunk_stats['chunk_label']}")
            self.result_text.append(f"  Frames: {chunk_stats['frames_count']}")
            self.result_text.append(f"  Points: {chunk_stats['total_points']:,}")
            self.result_text.append(f"  Tracks: {chunk_stats['total_tracks']:,}")
            self.result_text.append(f"  Projections: {chunk_stats['total_projections']:,}")
            self.result_text.append(f"  Cameras: {chunk_stats['cameras_count']}")
            self.result_text.append(f"  Metashape内存: {format_bytes(chunk_stats['estimated_memory'])}")
            
            # 🆕 显示export_colmap内存分析
            if 'export_memory_breakdown' in chunk_stats:
                breakdown = chunk_stats['export_memory_breakdown']
                self.result_text.append(f"  Export_colmap内存分析:")
                total_export = chunk_stats.get('total_export_colmap_memory', 0)
                self.result_text.append(f"    总额外内存: {format_bytes(total_export)}")
                self.result_text.append(f"    Tracks字典: {format_bytes(breakdown.get('total_tracks_dict_memory', 0))}")
                self.result_text.append(f"    Images字典: {format_bytes(breakdown.get('total_images_dict_memory', 0))}")
                self.result_text.append(f"    内存放大: {breakdown.get('memory_amplification_ratio', 0):.2f}x")
            
            # 显示前几个frame的信息
            if chunk_stats['frames_stats']:
                self.result_text.append("  Frame统计:")
                for i, frame_stats in enumerate(chunk_stats['frames_stats'][:3]):  # 只显示前3个
                    export_mem = frame_stats.get('export_colmap_memory', {})
                    export_total = export_mem.get('total_export_memory', 0)
                    self.result_text.append(f"    Frame {frame_stats['frame_index']}: "
                                          f"Points={frame_stats['points_count']:,}, "
                                          f"Tracks={frame_stats['tracks_count']:,}, "
                                          f"原始内存={format_bytes(frame_stats['estimated_memory'])}, "
                                          f"Export额外={format_bytes(export_total)}")
                if len(chunk_stats['frames_stats']) > 3:
                    self.result_text.append(f"    ... 还有 {len(chunk_stats['frames_stats']) - 3} 个frames")
            
            self.result_text.append("")

    def export_report(self):
        if not self.stats:
            return
        
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "保存统计报告", "metashape_stats_report.txt", "Text Files (*.txt)")
        
        if filename:
            if export_statistics_to_file(self.stats, filename):
                QtWidgets.QMessageBox.information(self, "成功", f"统计报告已保存到:\n{filename}")
            else:
                QtWidgets.QMessageBox.warning(self, "错误", "保存统计报告时出现错误")

def show_project_statistics():
    """显示项目统计对话框"""
    global app
    app = QtWidgets.QApplication.instance()
    parent = app.activeWindow()
    
    # 检查是否有打开的项目
    doc = Metashape.app.document
    if not doc or not hasattr(doc, 'chunks'):
        QtWidgets.QMessageBox.warning(None, "警告", "请先打开一个Metashape项目")
        return
    
    dialog = ProjectStatsGUI(parent)
    dialog.exec()

# 注册菜单项
label = "Scripts/Metashape项目统计分析 (含Export_colmap内存预测)"
Metashape.app.addMenuItem(label, show_project_statistics)
print(f"To execute this script press {label}")