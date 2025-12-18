import sys
import numpy as np
import cv2

from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import (
    QImage,
    QPixmap,
    QPainter,
    QPen,
    QBrush,
    QColor,
    QKeySequence,
)
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QToolBar,
    QAction,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QSlider,
    QComboBox,
    QMessageBox,
    QShortcut,
)

# =============================================================================
# NumPy <-> QImage helpers
# =============================================================================

def numpy_to_qimage_gray(arr: np.ndarray) -> QImage:
    """2D uint8 -> QImage Grayscale8"""
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    h, w = arr.shape
    qimg = QImage(arr.data, w, h, w, QImage.Format_Grayscale8)
    return qimg.copy()


def qimage_to_numpy_gray(img: QImage) -> np.ndarray:
    """QImage -> 2D uint8 (grayscale)"""
    img = img.convertToFormat(QImage.Format_Grayscale8)
    w = img.width()
    h = img.height()
    ptr = img.bits()
    ptr.setsize(img.bytesPerLine() * h)
    arr = np.frombuffer(ptr, np.uint8)
    arr = arr.reshape((h, img.bytesPerLine()))
    return arr[:, :w].copy()


def numpy_to_qimage_rgb(arr: np.ndarray) -> QImage:
    """H x W x 3 uint8 (RGB) -> QImage RGB888"""
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    h, w, ch = arr.shape
    assert ch == 3, "RGB array must have 3 channels"
    bytes_per_line = 3 * w
    qimg = QImage(arr.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return qimg.copy()


def qimage_to_numpy_rgb(img: QImage) -> np.ndarray:
    img = img.convertToFormat(QImage.Format_RGB888)

    w, h = img.width(), img.height()
    bytes_per_line = img.bytesPerLine()

    ptr = img.bits()
    ptr.setsize(h * bytes_per_line)

    arr = np.frombuffer(ptr, np.uint8)
    arr = arr.reshape((h, bytes_per_line))

    # ⚠️ відкидаємо padding
    arr = arr[:, :w * 3]

    return arr.reshape((h, w, 3)).copy()

# =============================================================================
# Zoomable view (Ctrl+wheel)
# =============================================================================

class ZoomableGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._zoom_factor = 1.0
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

    def wheelEvent(self, event):
        if QApplication.keyboardModifiers() == Qt.ControlModifier:
            factor = 1.1 if event.angleDelta().y() > 0 else (1 / 1.1)
            self._zoom_factor *= factor
            self.scale(factor, factor)
        else:
            super().wheelEvent(event)


# =============================================================================
# DFT drawing view
# =============================================================================

class DFTView(ZoomableGraphicsView):
    def __init__(self, editor, parent=None):
        super().__init__(parent)
        self.editor = editor
        self.setMouseTracking(True)
        self.drawing = False
        self.start_point = None
        self.last_point = None

    def map_to_image_point(self, event_pos):
        scene_pos = self.mapToScene(event_pos)
        if self.editor.dft_pixmap_item is None:
            return None

        pixmap_rect = self.editor.dft_pixmap_item.boundingRect()
        if not pixmap_rect.contains(scene_pos):
            return None

        return QPointF(scene_pos.x(), scene_pos.y())

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.editor.image_loaded:
            img_point = self.map_to_image_point(event.pos())
            if img_point is not None:
                self.drawing = True
                self.start_point = img_point
                self.last_point = img_point
                self.editor.begin_stroke()

                if self.editor.current_tool in ("pencil", "eraser"):
                    self.editor.paint_segment(self.last_point, img_point)
                    self.editor.update_from_mask()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.drawing and self.editor.image_loaded:
            img_point = self.map_to_image_point(event.pos())
            if img_point is not None:
                if self.editor.current_tool in ("pencil", "eraser"):
                    self.editor.paint_segment(self.last_point, img_point)
                    self.last_point = img_point
                    self.editor.update_from_mask()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing and self.editor.image_loaded:
            img_point = self.map_to_image_point(event.pos())
            if img_point is not None:
                tool = self.editor.current_tool
                if tool in ("pencil", "eraser"):
                    pass
                elif tool == "line":
                    self.editor.paint_line(self.start_point, img_point)
                elif tool == "circle":
                    self.editor.paint_circle(self.start_point, img_point, inverse=False)
                elif tool == "inv_circle":
                    self.editor.paint_circle(self.start_point, img_point, inverse=True)

                self.editor.update_from_mask()

            self.drawing = False
            self.start_point = None
            self.last_point = None
        super().mouseReleaseEvent(event)


# =============================================================================
# Main window
# =============================================================================

class FourierEditor(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Vitalik's periodic noise remover")
        self.resize(1200, 700)

        # State
        self.image_loaded = False

        self.original_rgb = None      # HxWx3 uint8 (RGB)
        self.original_gray = None     # HxW float32 (luma)
        self.dft_complex = None       # complex spectrum (shifted)
        self.mask_image = None        # QImage grayscale mask (0..255)

        self.undo_stack = []
        self.redo_stack = []

        self.current_tool = "pencil"
        self.brush_thickness = 5
        self.brush_intensity = 0  # 0 black (strong suppress), 255 white (no change)

        self.last_result_rgb = None  # store latest processed RGB (uint8)

        self.setAcceptDrops(True)

        self._create_toolbar()
        self._create_central_widgets()
        self._create_shortcuts()

    # -------------------------------------------------------------------------
    # Toolbar (IMPORTANT: shortcuts live here to avoid "Ambiguous shortcut overload")
    # -------------------------------------------------------------------------

    def _create_toolbar(self):
        toolbar = QToolBar("Main Toolbar", self)
        self.addToolBar(toolbar)

        open_action = QAction("Відкрити", self)
        open_action.setShortcut(QKeySequence("Ctrl+O"))
        open_action.triggered.connect(self.open_image)
        toolbar.addAction(open_action)

        save_action = QAction("Зберегти  ", self)
        save_action.setShortcut(QKeySequence("Ctrl+S"))
        save_action.triggered.connect(self.save_image)
        toolbar.addAction(save_action)

        toolbar.addSeparator()

        undo_action = QAction("Undo", self)
        undo_action.setShortcut(QKeySequence("Ctrl+Z"))
        undo_action.triggered.connect(self.undo)
        toolbar.addAction(undo_action)

        redo_action = QAction("Redo", self)
        redo_action.setShortcut(QKeySequence("Ctrl+Y"))
        redo_action.triggered.connect(self.redo)
        toolbar.addAction(redo_action)

        reset_action = QAction("Відновити", self)
        reset_action.triggered.connect(self.reset_mask)
        toolbar.addAction(reset_action)

        toolbar.addSeparator()

        self.tool_combo = QComboBox()
        self.tool_combo.addItem("Олівець", "pencil")
        self.tool_combo.addItem("Лінія", "line")
        self.tool_combo.addItem("Круг", "circle")
        self.tool_combo.addItem("Зворотній круг", "inv_circle")
        self.tool_combo.addItem("Гумка", "eraser")
        self.tool_combo.currentIndexChanged.connect(self._tool_changed)
        toolbar.addWidget(QLabel(" Інструмент: "))
        toolbar.addWidget(self.tool_combo)

        self.thickness_slider = QSlider(Qt.Horizontal)
        self.thickness_slider.setMinimum(1)
        self.thickness_slider.setMaximum(50)
        self.thickness_slider.setValue(self.brush_thickness)
        self.thickness_slider.valueChanged.connect(self._thickness_changed)
        toolbar.addWidget(QLabel("  Товщина: "))
        toolbar.addWidget(self.thickness_slider)

        self.intensity_slider = QSlider(Qt.Horizontal)
        self.intensity_slider.setMinimum(0)
        self.intensity_slider.setMaximum(255)
        self.intensity_slider.setValue(self.brush_intensity)
        self.intensity_slider.valueChanged.connect(self._intensity_changed)
        toolbar.addWidget(QLabel("  Непрозорість: "))
        toolbar.addWidget(self.intensity_slider)

    # -------------------------------------------------------------------------
    # Central UI
    # -------------------------------------------------------------------------

    def _create_central_widgets(self):
        central = QWidget()
        self.setCentralWidget(central)

        layout = QHBoxLayout()
        central.setLayout(layout)

        left_layout = QVBoxLayout()
        layout.addLayout(left_layout)

        lbl_dft = QLabel("Спектр Фур'є:")
        left_layout.addWidget(lbl_dft)

        self.dft_scene = QGraphicsScene(self)
        self.dft_view = DFTView(self)
        self.dft_view.setScene(self.dft_scene)
        left_layout.addWidget(self.dft_view)

        right_layout = QVBoxLayout()
        layout.addLayout(right_layout)

        lbl_img = QLabel("Результат:")
        right_layout.addWidget(lbl_img)

        self.img_scene = QGraphicsScene(self)
        self.img_view = ZoomableGraphicsView(self)
        self.img_view.setScene(self.img_scene)
        right_layout.addWidget(self.img_view)

        self.dft_pixmap_item = None
        self.img_pixmap_item = None

    # -------------------------------------------------------------------------
    # Ctrl+C, Ctrl+V
    # -------------------------------------------------------------------------

    def _create_shortcuts(self):
        # Clipboard
        QShortcut(QKeySequence("Ctrl+V"), self, activated=self.paste_image)
        QShortcut(QKeySequence("Ctrl+C"), self, activated=self.copy_image)

    # -------------------------------------------------------------------------
    # Tool params
    # -------------------------------------------------------------------------

    def _tool_changed(self, index):
        self.current_tool = self.tool_combo.itemData(index)

    def _thickness_changed(self, value):
        self.brush_thickness = value

    def _intensity_changed(self, value):
        self.brush_intensity = value

    # -------------------------------------------------------------------------
    # Open / Load
    # -------------------------------------------------------------------------

    def open_image(self):
        fname, _ = QFileDialog.getOpenFileName(
            self,
            "Виберіть зображення",
            "",
            "Формат зображення (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)",
        )
        if not fname:
            return

        img_bgr = cv2.imread(fname, cv2.IMREAD_COLOR)
        if img_bgr is None:
            QMessageBox.warning(self, "Помилка", "Не вдалося відкрити зображення.")
            return

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self.load_new_image(img_rgb)

    def load_new_image(self, rgb_arr: np.ndarray):
        """Завантажте RGB зображення. """
        if rgb_arr is None:
            return
        if rgb_arr.ndim != 3 or rgb_arr.shape[2] != 3:
            QMessageBox.warning(self, "Помилка", "Очікувалося кольорове зображення (RGB).")
            return

        self.original_rgb = rgb_arr.astype(np.uint8)

        gray = cv2.cvtColor(self.original_rgb, cv2.COLOR_RGB2GRAY)
        self.original_gray = gray.astype(np.float32)

        self.image_loaded = True
        self.compute_dft()

        h, w = gray.shape
        mask_np = np.full((h, w), 255, dtype=np.uint8)
        self.mask_image = numpy_to_qimage_gray(mask_np)

        self.undo_stack.clear()
        self.redo_stack.clear()

        self.update_from_mask()

    # -------------------------------------------------------------------------
    # DFT / Update
    # -------------------------------------------------------------------------

    def compute_dft(self):
        f = np.fft.fft2(self.original_gray)
        self.dft_complex = np.fft.fftshift(f)

    def update_from_mask(self):
        if not self.image_loaded or self.dft_complex is None or self.mask_image is None:
            return

        mask_np = qimage_to_numpy_gray(self.mask_image).astype(np.float32) / 255.0

        # Mask spectrum (luminance spectrum)
        F_mod = self.dft_complex * mask_np

        # DFT view (log magnitude)
        magnitude = np.abs(F_mod)
        magnitude = np.log(1 + magnitude)
        magnitude -= magnitude.min()
        if magnitude.max() > 0:
            magnitude = magnitude / magnitude.max()
        dft_display = (magnitude * 255).astype(np.uint8)

        dft_qimage = numpy_to_qimage_gray(dft_display)
        dft_pixmap = QPixmap.fromImage(dft_qimage)
        if self.dft_pixmap_item is None:
            self.dft_pixmap_item = QGraphicsPixmapItem(dft_pixmap)
            self.dft_scene.addItem(self.dft_pixmap_item)
        else:
            self.dft_pixmap_item.setPixmap(dft_pixmap)

        # Reconstruct luminance
        lum = np.fft.ifft2(np.fft.ifftshift(F_mod))
        lum = np.real(lum)

        # Normalize to [0..255]
        lum -= lum.min()
        if lum.max() > 0:
            lum = lum / lum.max()
        lum_u8 = (lum * 255).astype(np.uint8)

        # Apply luminance ratio to original RGB to keep colors
        lum_orig = cv2.cvtColor(self.original_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        lum_new = lum_u8.astype(np.float32)

        ratio = lum_new / (lum_orig + 1e-6)
        ratio = np.clip(ratio, 0.0, 10.0)  # avoid extreme blow-ups

        result = self.original_rgb.astype(np.float32).copy()
        for c in range(3):
            result[:, :, c] = np.clip(result[:, :, c] * ratio, 0, 255)
        result_u8 = result.astype(np.uint8)

        self.last_result_rgb = result_u8

        img_qimage = numpy_to_qimage_rgb(result_u8)
        img_pixmap = QPixmap.fromImage(img_qimage)

        if self.img_pixmap_item is None:
            self.img_pixmap_item = QGraphicsPixmapItem(img_pixmap)
            self.img_scene.addItem(self.img_pixmap_item)
        else:
            self.img_pixmap_item.setPixmap(img_pixmap)

    # -------------------------------------------------------------------------
    # Drawing on mask
    # -------------------------------------------------------------------------

    def begin_stroke(self):
        if self.mask_image is None:
            return
        self.undo_stack.append(self.mask_image.copy())
        self.redo_stack.clear()

    def get_pen_color(self):
        intensity = 255 if self.current_tool == "eraser" else self.brush_intensity
        return QColor(intensity, intensity, intensity)

    def paint_segment(self, p1: QPointF, p2: QPointF):
        if self.mask_image is None:
            return
        painter = QPainter(self.mask_image)
        pen = QPen(
            self.get_pen_color(),
            self.brush_thickness,
            Qt.SolidLine,
            Qt.RoundCap,
            Qt.RoundJoin,
        )
        painter.setPen(pen)
        painter.drawLine(p1, p2)
        painter.end()

    def paint_line(self, start: QPointF, end: QPointF):
        if self.mask_image is None:
            return
        painter = QPainter(self.mask_image)
        pen = QPen(
            self.get_pen_color(),
            self.brush_thickness,
            Qt.SolidLine,
            Qt.RoundCap,
            Qt.RoundJoin,
        )
        painter.setPen(pen)
        painter.drawLine(start, end)
        painter.end()

    def paint_circle(self, start: QPointF, end: QPointF, inverse=False):
        if self.mask_image is None:
            return

        x1, y1 = int(start.x()), int(start.y())
        x2, y2 = int(end.x()), int(end.y())

        left = min(x1, x2)
        right = max(x1, x2)
        top = min(y1, y2)
        bottom = max(y1, y2)

        w = right - left
        h = bottom - top

        painter = QPainter(self.mask_image)

        if not inverse:
            pen_color = self.get_pen_color()
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(pen_color, Qt.SolidPattern))
            painter.drawEllipse(left, top, w, h)
        else:
            dark_color = self.get_pen_color()  # intensity from slider
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(dark_color, Qt.SolidPattern))
            painter.drawRect(0, 0, self.mask_image.width(), self.mask_image.height())

            painter.setBrush(QBrush(QColor(255, 255, 255), Qt.SolidPattern))
            painter.drawEllipse(left, top, w, h)

        painter.end()

    # -------------------------------------------------------------------------
    # Undo / Redo / Reset
    # -------------------------------------------------------------------------

    def undo(self):
        if not self.image_loaded or not self.undo_stack:
            return
        self.redo_stack.append(self.mask_image.copy())
        self.mask_image = self.undo_stack.pop()
        self.update_from_mask()

    def redo(self):
        if not self.image_loaded or not self.redo_stack:
            return
        self.undo_stack.append(self.mask_image.copy())
        self.mask_image = self.redo_stack.pop()
        self.update_from_mask()

    def reset_mask(self):
        if not self.image_loaded:
            return
        h, w = self.original_gray.shape
        mask_np = np.full((h, w), 255, dtype=np.uint8)
        self.mask_image = numpy_to_qimage_gray(mask_np)
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.update_from_mask()

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    def save_image(self):
        if not self.image_loaded or self.last_result_rgb is None:
            QMessageBox.information(self, "Зберегти", "Немає що зберігати.")
            return

        fname, _ = QFileDialog.getSaveFileName(
            self,
            "Зберегти оброблене зображення",
            "",
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp);;TIFF (*.tif *.tiff);;WEBP (*.webp)",
        )
        if not fname:
            return

        # Save via OpenCV (needs BGR)
        bgr = cv2.cvtColor(self.last_result_rgb, cv2.COLOR_RGB2BGR)
        ok = cv2.imwrite(fname, bgr)
        if not ok:
            QMessageBox.warning(self, "Помилка", "Не вдалося зберегти файл.")

    # -------------------------------------------------------------------------
    # Clipboard
    # -------------------------------------------------------------------------

    def paste_image(self):
        clipboard = QApplication.clipboard()
        img = clipboard.image()
        if img.isNull():
            return

        rgb = qimage_to_numpy_rgb(img)
        self.load_new_image(rgb)

    def copy_image(self):
        if self.img_pixmap_item is None:
            return
        pixmap = self.img_pixmap_item.pixmap()
        QApplication.clipboard().setPixmap(pixmap)

# =============================================================================
# Run
# =============================================================================

def main():
    app = QApplication(sys.argv)
    win = FourierEditor()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
