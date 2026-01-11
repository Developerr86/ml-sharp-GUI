"""
SHARP GUI Application
Frontend for the Apple SHARP 3D Gaussian Splatting model.
"""

import sys
import logging
from pathlib import Path

# Add src to sys.path to allow importing sharp
current_dir = Path(__file__).parent.resolve()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Configure logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

import torch
import numpy as np
import torch.nn.functional as F

# PyQt6 imports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QPushButton, QLabel, QSlider, QFileDialog, QDockWidget,
    QMessageBox, QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QImage, QPixmap, QAction, QMouseEvent, QWheelEvent

# Sharp imports
from sharp.models import PredictorParams, create_predictor
from sharp.utils import io
from sharp.utils.gaussians import (
    Gaussians3D,
    SceneMetaData,
    unproject_gaussians,
    save_ply,
    load_ply
)
from sharp.utils.camera import create_camera_matrix
from sharp.utils.gsplat import GSplatRenderer

# Constants
DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
INTERNAL_SHAPE = (1536, 1536)


class InferenceSignals(QObject):
    """Signals for the inference worker thread."""
    finished = pyqtSignal(object, object)  # gaussians, metadata
    error = pyqtSignal(str)
    status = pyqtSignal(str)


class InferenceWorker(QThread):
    """Worker thread for model inference."""
    
    def __init__(self, image_path: Path, device: str):
        super().__init__()
        self.image_path = image_path
        self.device = device
        self.signals = InferenceSignals()

    def run(self):
        try:
            self.signals.status.emit("Loading Model...")
            
            # Check/Download checkpoint
            checkpoint_path = Path("sharp_2572gikvuh.pt")
            if not checkpoint_path.exists():
                self.signals.status.emit("Downloading Checkpoint...")
                LOGGER.info("Downloading default model from %s", DEFAULT_MODEL_URL)
                state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)
            else:
                LOGGER.info("Loading checkpoint from %s", checkpoint_path)
                state_dict = torch.load(checkpoint_path, weights_only=True)
            
            self.signals.status.emit("Initializing Predictor...")
            gaussian_predictor = create_predictor(PredictorParams())
            gaussian_predictor.load_state_dict(state_dict)
            gaussian_predictor.eval()
            gaussian_predictor.to(self.device)
            
            self.signals.status.emit("Processing Image...")
            image, _, f_px = io.load_rgb(self.image_path)
            height, width = image.shape[:2]
            
            LOGGER.info("Running preprocessing.")
            image_pt = torch.from_numpy(image.copy()).float().to(self.device).permute(2, 0, 1) / 255.0
            disparity_factor = torch.tensor([f_px / width]).float().to(self.device)
            
            image_resized_pt = F.interpolate(
                image_pt[None],
                size=(INTERNAL_SHAPE[1], INTERNAL_SHAPE[0]),
                mode="bilinear",
                align_corners=True,
            )
            
            self.signals.status.emit("Running Inference...")
            with torch.no_grad():
                gaussians_ndc = gaussian_predictor(image_resized_pt, disparity_factor)
                
                self.signals.status.emit("Unprojecting...")
                intrinsics = torch.tensor([
                    [f_px, 0, width / 2, 0],
                    [0, f_px, height / 2, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]).float().to(self.device)
                
                intrinsics_resized = intrinsics.clone()
                intrinsics_resized[0] *= INTERNAL_SHAPE[0] / width
                intrinsics_resized[1] *= INTERNAL_SHAPE[1] / height
                
                gaussians = unproject_gaussians(
                    gaussians_ndc, torch.eye(4).to(self.device), intrinsics_resized, INTERNAL_SHAPE
                )
            
            metadata = SceneMetaData(f_px, (width, height), "linearRGB")
            self.signals.finished.emit(gaussians, metadata)
            
        except Exception as e:
            LOGGER.error("Inference failed", exc_info=True)
            self.signals.error.emit(str(e))


class OrbitCamera:
    """Simple orbit camera for 3D navigation."""
    
    def __init__(self, radius=3.0, azimuth=0.0, elevation=0.0):
        self.radius = radius
        self.azimuth = azimuth
        self.elevation = elevation
        self.center = torch.zeros(3)

    def get_extrinsics(self, device):
        # Convert spherical to cartesian
        x = self.radius * np.cos(self.elevation) * np.sin(self.azimuth)
        y = self.radius * np.sin(self.elevation)
        z = self.radius * np.cos(self.elevation) * np.cos(self.azimuth)
        
        position = torch.tensor([x, y, z], device=device).float()
        look_at = self.center.to(device)
        up = torch.tensor([0.0, 1.0, 0.0], device=device)
        
        return create_camera_matrix(position, look_at, up, inverse=True)


class RenderArea(QLabel):
    """Widget to display the rendered 3D Gaussians."""
    
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: #1a1a1a; color: #888888;")
        self.setText("Render Area\n(Load Image to Start)")
        self.setMinimumSize(640, 480)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        
        self.renderer = GSplatRenderer(background_color="black")
        self.camera = OrbitCamera(radius=4.0)
        self.gaussians = None
        self.metadata = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.last_mouse_pos = None
        self.splat_scale = 1.0
        
    def set_scene(self, gaussians: Gaussians3D, metadata: SceneMetaData):
        self.gaussians = gaussians
        self.metadata = metadata
        self.setText("")
        self.render_frame()

    def render_frame(self):
        if self.gaussians is None:
            return

        w, h = self.width(), self.height()
        if w <= 0 or h <= 0:
            return
        
        # Adjust intrinsics for current viewport
        orig_w, orig_h = self.metadata.resolution_px
        f_px = self.metadata.focal_length_px
        
        scale = w / orig_w
        current_f = f_px * scale
        
        intrinsics = torch.tensor([
            [current_f, 0, w / 2, 0],
            [0, current_f, h / 2, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], device=self.device).float()
        
        extrinsics = self.camera.get_extrinsics(self.device)
        
        # Apply splat scale
        gaussians_scaled = self.gaussians
        if self.splat_scale != 1.0:
            gaussians_scaled = Gaussians3D(
                mean_vectors=self.gaussians.mean_vectors,
                singular_values=self.gaussians.singular_values * self.splat_scale,
                quaternions=self.gaussians.quaternions,
                colors=self.gaussians.colors,
                opacities=self.gaussians.opacities
            )

        with torch.no_grad():
            output = self.renderer(
                gaussians_scaled,
                extrinsics.unsqueeze(0),
                intrinsics.unsqueeze(0),
                w, h
            )
            
        # Convert to QImage
        color_np = (output.color.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        img_h, img_w, c = color_np.shape
        bytes_per_line = c * img_w
        qimg = QImage(color_np.tobytes(), img_w, img_h, bytes_per_line, QImage.Format.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qimg))
        
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_mouse_pos = event.position()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.last_mouse_pos and (event.buttons() & Qt.MouseButton.LeftButton):
            delta = event.position() - self.last_mouse_pos
            self.last_mouse_pos = event.position()
            
            sensitivity = 0.01
            self.camera.azimuth -= delta.x() * sensitivity
            self.camera.elevation = np.clip(
                self.camera.elevation + delta.y() * sensitivity,
                -np.pi/2 + 0.1, np.pi/2 - 0.1
            )
            self.render_frame()

    def mouseReleaseEvent(self, event: QMouseEvent):
        self.last_mouse_pos = None

    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y()
        self.camera.radius *= (0.9 if delta > 0 else 1.1)
        self.render_frame()
        
    def set_radius(self, radius):
        self.camera.radius = radius
        self.render_frame()

    def set_splat_scale(self, scale):
        self.splat_scale = scale
        self.render_frame()


class ControlPanel(QWidget):
    """Right-side control panel."""
    
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        
        # Status Label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
        layout.addWidget(self.status_label)
        
        layout.addSpacing(20)
        
        # Generate Button
        self.generate_btn = QPushButton("Generate 3D Model")
        self.generate_btn.setEnabled(False)
        self.generate_btn.setMinimumHeight(40)
        layout.addWidget(self.generate_btn)
        
        layout.addSpacing(20)
        
        # Camera Radius Slider
        layout.addWidget(QLabel("Camera Radius (Zoom)"))
        self.radius_slider = QSlider(Qt.Orientation.Horizontal)
        self.radius_slider.setRange(10, 200)
        self.radius_slider.setValue(40)
        layout.addWidget(self.radius_slider)
        
        # Splat Scale Slider
        layout.addWidget(QLabel("Splat Scale"))
        self.scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.scale_slider.setRange(10, 200)
        self.scale_slider.setValue(100)
        layout.addWidget(self.scale_slider)
        
        layout.addStretch()
        self.setLayout(layout)


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SHARP 3D Gaussian Splatting Viewer")
        self.resize(1200, 800)
        
        # Central Widget (Render Area)
        self.render_area = RenderArea()
        self.setCentralWidget(self.render_area)
        
        # Dock Widget (Control Panel)
        self.dock = QDockWidget("Controls", self)
        self.dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea)
        self.control_panel = ControlPanel()
        self.dock.setWidget(self.control_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock)
        
        # Menu Bar
        self._create_menus()
        
        # Connect signals
        self.control_panel.generate_btn.clicked.connect(self.on_generate_clicked)
        self.control_panel.radius_slider.valueChanged.connect(self.on_radius_changed)
        self.control_panel.scale_slider.valueChanged.connect(self.on_scale_changed)
        
        # State
        self.current_image_path = None
        self.gaussians = None
        self.metadata = None
        self.worker = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.device != "cuda":
            QMessageBox.warning(self, "Device Warning", 
                "CUDA not detected. Rendering might be slow or fail if gsplat requires CUDA.")

    def _create_menus(self):
        menu = self.menuBar()
        file_menu = menu.addMenu("File")
        
        open_img_action = QAction("Open Image...", self)
        open_img_action.triggered.connect(self.open_image)
        file_menu.addAction(open_img_action)
        
        open_ply_action = QAction("Open PLY...", self)
        open_ply_action.triggered.connect(self.open_ply)
        file_menu.addAction(open_ply_action)
        
        save_ply_action = QAction("Save PLY...", self)
        save_ply_action.triggered.connect(self.save_ply)
        file_menu.addAction(save_ply_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def open_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.heic)"
        )
        if file_name:
            self.current_image_path = Path(file_name)
            self.control_panel.status_label.setText(f"Loaded: {self.current_image_path.name}")
            self.control_panel.generate_btn.setEnabled(True)
            self.render_area.setText(f"Image Loaded: {self.current_image_path.name}\nReady to Generate.")
            
    def open_ply(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open PLY", "", "PLY Files (*.ply)")
        if file_name:
            try:
                gaussians, metadata = load_ply(Path(file_name))
                self.gaussians = gaussians.to(self.device)
                self.metadata = metadata
                self.render_area.set_scene(self.gaussians, self.metadata)
                self.control_panel.status_label.setText(f"Loaded PLY: {Path(file_name).name}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load PLY: {e}")

    def save_ply(self):
        if self.gaussians is None or self.metadata is None:
            QMessageBox.warning(self, "Warning", "No model to save. Generate or load a model first.")
            return
        file_name, _ = QFileDialog.getSaveFileName(self, "Save PLY", "", "PLY Files (*.ply)")
        if file_name:
            try:
                save_ply(
                    self.gaussians, 
                    self.metadata.focal_length_px, 
                    self.metadata.resolution_px, 
                    Path(file_name)
                )
                self.control_panel.status_label.setText("Saved PLY")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save PLY: {e}")

    def on_generate_clicked(self):
        if not self.current_image_path:
            return
        
        self.control_panel.status_label.setText("Starting Worker...")
        self.control_panel.generate_btn.setEnabled(False)
        
        self.worker = InferenceWorker(self.current_image_path, self.device)
        self.worker.signals.status.connect(self.update_status)
        self.worker.signals.finished.connect(self.on_inference_finished)
        self.worker.signals.error.connect(self.on_inference_error)
        self.worker.start()

    def update_status(self, msg):
        self.control_panel.status_label.setText(msg)

    def on_inference_finished(self, gaussians, metadata):
        self.gaussians = gaussians
        self.metadata = metadata
        self.control_panel.status_label.setText("Generation Complete")
        self.control_panel.generate_btn.setEnabled(True)
        self.render_area.set_scene(gaussians, metadata)
        
    def on_inference_error(self, err_msg):
        self.control_panel.status_label.setText("Error")
        self.control_panel.generate_btn.setEnabled(True)
        QMessageBox.critical(self, "Inference Error", f"An error occurred:\n{err_msg}")

    def on_radius_changed(self, value):
        radius = value / 10.0
        self.render_area.set_radius(radius)

    def on_scale_changed(self, value):
        scale = value / 100.0
        self.render_area.set_splat_scale(scale)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
