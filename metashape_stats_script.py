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

def analyze_frame_statistics(frame):
    """分析单个frame的统计信息"""
    stats = {
        'points_count': 0,
        'tracks_count': 0,
        'projections_count': 0,
        'cameras_with_projections': 0,
        'estimated_memory': 0
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
            
            # 估算内存使用
            stats['estimated_memory'] = get_tie_points_memory_estimate(tie_points)
    
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
        'frames_stats': []
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
            
            for i, frame in enumerate(chunk.frames):
                frame_stats = analyze_frame_statistics(frame)
                frame_stats['frame_index'] = i
                stats['frames_stats'].append(frame_stats)
                
                # 累计统计
                stats['total_points'] += frame_stats['points_count']
                stats['total_tracks'] += frame_stats['tracks_count']
                stats['total_projections'] += frame_stats['projections_count']
                stats['estimated_memory'] += frame_stats['estimated_memory']
    
    except Exception as e:
        print(f"分析chunk时出错: {e}")
    
    return stats

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
        'memory_info': initial_memory
    }
    
    try:
        # 分析chunks
        if hasattr(doc, 'chunks') and doc.chunks:
            stats['chunks_count'] = len(doc.chunks)
            
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
    
    except Exception as e:
        print(f"分析document时出错: {e}")
    
    return stats

def export_statistics_to_file(stats, filepath):
    """将统计信息导出到文件"""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("Metashape项目统计报告\n")
            f.write("=" * 50 + "\n\n")
            
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
            f.write(f"  估算内存使用: {format_bytes(stats['estimated_memory'])}\n\n")
            
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
                f.write(f"  估算内存: {format_bytes(chunk_stats['estimated_memory'])}\n")
                
                # Frame详细信息
                if chunk_stats['frames_stats']:
                    f.write("  Frame详情:\n")
                    for frame_stats in chunk_stats['frames_stats']:
                        f.write(f"    Frame {frame_stats['frame_index']}: ")
                        f.write(f"Points={frame_stats['points_count']:,}, ")
                        f.write(f"Tracks={frame_stats['tracks_count']:,}, ")
                        f.write(f"Projections={frame_stats['projections_count']:,}, ")
                        f.write(f"Memory={format_bytes(frame_stats['estimated_memory'])}\n")
                f.write("\n")
        
        return True
    except Exception as e:
        print(f"导出统计信息时出错: {e}")
        return False

class ProjectStatsGUI(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(ProjectStatsGUI, self).__init__(parent)
        self.setWindowTitle("Metashape项目统计分析")
        self.setMinimumSize(800, 600)
        self.stats = None
        self.setup_ui()

    def setup_ui(self):
        layout = QtWidgets.QVBoxLayout()
        
        # 标题
        title_label = QtWidgets.QLabel("Metashape项目统计分析")
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        layout.addWidget(title_label)
        
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
        self.result_text.setFont(QtGui.QFont("Consolas", 10))
        layout.addWidget(self.result_text)
        
        self.setLayout(layout)

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
        self.result_text.append(f"估算内存使用: {format_bytes(self.stats['estimated_memory'])}")
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
            self.result_text.append(f"  估算内存: {format_bytes(chunk_stats['estimated_memory'])}")
            
            # 显示前几个frame的信息
            if chunk_stats['frames_stats']:
                self.result_text.append("  Frame统计:")
                for i, frame_stats in enumerate(chunk_stats['frames_stats'][:5]):  # 只显示前5个
                    self.result_text.append(f"    Frame {frame_stats['frame_index']}: "
                                          f"Points={frame_stats['points_count']:,}, "
                                          f"Tracks={frame_stats['tracks_count']:,}, "
                                          f"Memory={format_bytes(frame_stats['estimated_memory'])}")
                if len(chunk_stats['frames_stats']) > 5:
                    self.result_text.append(f"    ... 还有 {len(chunk_stats['frames_stats']) - 5} 个frames")
            
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
label = "Scripts/Metashape项目统计分析"
Metashape.app.addMenuItem(label, show_project_statistics)
print(f"To execute this script press {label}")
