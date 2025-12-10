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


# ===== Допоміжні функції для NumPy <-> QImage =====

def numpy_to_qimage_gray(arr: np.ndarray) -> QImage:
    """
    Перетворює 2D NumPy (uint8) у QImage (Format_Grayscale8).
    """
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    h, w = arr.shape
    qimg = QImage(arr.data, w, h, w, QImage.Format_Grayscale8)
    return qimg.copy()


def qimage_to_numpy_gray(img: QImage) -> np.ndarray:
    """
    Перетворює QImage (Format_Grayscale8) у 2D NumPy (uint8).
    """
    img = img.convertToFormat(QImage.Format_Grayscale8)
    w = img.width()
    h = img.height()
    ptr = img.bits()
    ptr.setsize(img.bytesPerLine() * h)
    arr = np.frombuffer(ptr, np.uint8)
    arr = arr.reshape((h, img.bytesPerLine()))
    return arr[:, :w].copy()


# ===== ZoomableGraphicsView (Ctrl + колесо = zoom) =====

class ZoomableGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._zoom_factor = 1.0
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

    def wheelEvent(self, event):
        if QApplication.keyboardModifiers() == Qt.ControlModifier:
            # Масштабування
            if event.angleDelta().y() > 0:
                factor = 1.1
            else:
                factor = 1 / 1.1
            self._zoom_factor *= factor
            self.scale(factor, factor)
        else:
            super().wheelEvent(event)


# ===== Вікно DFT, по якому малюємо =====

class DFTView(ZoomableGraphicsView):
    def __init__(self, editor, parent=None):
        super().__init__(parent)
        self.editor = editor
        self.setMouseTracking(True)
        self.drawing = False
        self.start_point = None
        self.last_point = None

    def map_to_image_point(self, event_pos):
        """
        Переводимо координати миші в координати зображення в сцені.
        """
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
                self.editor.begin_stroke()  # зберігаємо маску для Undo

                # Олівець / ластик – малюємо одразу
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
                    # все вже домальовано по ходу руху
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


# ===== Головне вікно FourierEditor =====

class FourierEditor(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Vitalik's periodic noise remover")
        self.resize(1200, 700)

        # Стан
        self.image_loaded = False
        self.original_image = None    # NumPy, grayscale, float32
        self.dft_complex = None       # комплексний спектр
        self.mask_image = None        # QImage (маска спектра)

        self.undo_stack = []          # список масок (QImage)
        self.redo_stack = []

        self.current_tool = "pencil"  # pencil | line | circle | inv_circle | eraser
        self.brush_thickness = 5
        self.brush_intensity = 0      # 0 – чорний, 255 – білий

        # Приймаємо drop
        self.setAcceptDrops(True)

        # Інтерфейс
        self._create_toolbar()
        self._create_central_widgets()
        self._create_shortcuts()

    # ---------- Toolbar ----------

    def _create_toolbar(self):
        toolbar = QToolBar("Main Toolbar", self)
        self.addToolBar(toolbar)

        # Open
        open_action = QAction("Open", self)
        open_action.setShortcut(QKeySequence("Ctrl+O"))
        open_action.triggered.connect(self.open_image)
        toolbar.addAction(open_action)

        # Undo / Redo
        undo_action = QAction("Undo", self)
        undo_action.setShortcut(QKeySequence("Ctrl+Z"))
        undo_action.triggered.connect(self.undo)
        toolbar.addAction(undo_action)

        redo_action = QAction("Redo", self)
        redo_action.setShortcut(QKeySequence("Ctrl+Y"))
        redo_action.triggered.connect(self.redo)
        toolbar.addAction(redo_action)

        reset_action = QAction("Reset", self)
        reset_action.triggered.connect(self.reset_mask)
        toolbar.addAction(reset_action)

        toolbar.addSeparator()

        # Вибір інструмента
        self.tool_combo = QComboBox()
        self.tool_combo.addItem("Pencil", "pencil")
        self.tool_combo.addItem("Line", "line")
        self.tool_combo.addItem("Filled circle", "circle")
        self.tool_combo.addItem("Inverse circle", "inv_circle")
        self.tool_combo.addItem("Eraser", "eraser")
        self.tool_combo.currentIndexChanged.connect(self._tool_changed)
        toolbar.addWidget(QLabel(" Tool: "))
        toolbar.addWidget(self.tool_combo)

        # Товщина
        self.thickness_slider = QSlider(Qt.Horizontal)
        self.thickness_slider.setMinimum(1)
        self.thickness_slider.setMaximum(50)
        self.thickness_slider.setValue(self.brush_thickness)
        self.thickness_slider.valueChanged.connect(self._thickness_changed)
        toolbar.addWidget(QLabel("  Thickness: "))
        toolbar.addWidget(self.thickness_slider)

        # Відтінок чорного
        # 0   = чорний   (максимальне приглушення)
        # 255 = білий    (без змін)
        self.intensity_slider = QSlider(Qt.Horizontal)
        self.intensity_slider.setMinimum(0)
        self.intensity_slider.setMaximum(255)
        self.intensity_slider.setValue(self.brush_intensity)
        self.intensity_slider.valueChanged.connect(self._intensity_changed)
        toolbar.addWidget(QLabel("  Shade (0=black, 255=white): "))
        toolbar.addWidget(self.intensity_slider)

    # ---------- Центральна область (дві сцени) ----------

    def _create_central_widgets(self):
        central = QWidget()
        self.setCentralWidget(central)

        layout = QHBoxLayout()
        central.setLayout(layout)

        # Ліва частина – DFT
        left_layout = QVBoxLayout()
        layout.addLayout(left_layout)

        lbl_dft = QLabel("DFT (спектр, по якому ми малюємо)")
        left_layout.addWidget(lbl_dft)

        self.dft_scene = QGraphicsScene(self)
        self.dft_view = DFTView(self)
        self.dft_view.setScene(self.dft_scene)
        left_layout.addWidget(self.dft_view)

        # Права частина – відновлене зображення
        right_layout = QVBoxLayout()
        layout.addLayout(right_layout)

        lbl_img = QLabel("Відновлене зображення після змін у DFT")
        right_layout.addWidget(lbl_img)

        self.img_scene = QGraphicsScene(self)
        self.img_view = ZoomableGraphicsView(self)
        self.img_view.setScene(self.img_scene)
        right_layout.addWidget(self.img_view)

        # Елементи сцени
        self.dft_pixmap_item = None
        self.img_pixmap_item = None

    # ---------- Шорткати ----------

    def _create_shortcuts(self):
        # Undo
        QShortcut(QKeySequence("Ctrl+Z"), self, activated=self.undo)
        # Redo класичний
        QShortcut(QKeySequence("Ctrl+Y"), self, activated=self.redo)
        # Redo як у Photoshop
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self, activated=self.redo)

        # Paste image з буфера
        QShortcut(QKeySequence("Ctrl+V"), self, activated=self.paste_image)
        # Copy image в буфер
        QShortcut(QKeySequence("Ctrl+C"), self, activated=self.copy_image)

    # ---------- Зміна інструментів/параметрів ----------

    def _tool_changed(self, index):
        self.current_tool = self.tool_combo.itemData(index)

    def _thickness_changed(self, value):
        self.brush_thickness = value

    def _intensity_changed(self, value):
        self.brush_intensity = value

    # ---------- Відкриття / завантаження зображення ----------

    def open_image(self):
        fname, _ = QFileDialog.getOpenFileName(
            self,
            "Виберіть зображення",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        if not fname:
            return

        img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        if img is None:
            QMessageBox.warning(self, "Помилка", "Не вдалося відкрити зображення.")
            return

        self.load_new_image(img)

    def load_new_image(self, arr: np.ndarray):
        """
        Завантажити нове зображення (NumPy grayscale) в редактор.
        """
        self.original_image = arr.astype(np.float32)
        self.image_loaded = True

        self.compute_dft()

        h, w = arr.shape
        mask_np = np.full((h, w), 255, dtype=np.uint8)
        self.mask_image = numpy_to_qimage_gray(mask_np)

        self.undo_stack.clear()
        self.redo_stack.clear()

        self.update_from_mask()

    # ---------- DFT / оновлення вікон ----------

    def compute_dft(self):
        """
        Обчислюємо DFT зображення та зсуваємо нульові частоти в центр.
        """
        f = np.fft.fft2(self.original_image)
        fshift = np.fft.fftshift(f)
        self.dft_complex = fshift

    def update_from_mask(self):
        """
        Оновлюємо:
        - вікно DFT (лог-модуль спектра після маскування)
        - відновлене зображення (IFFT).
        """
        if not self.image_loaded or self.dft_complex is None or self.mask_image is None:
            return

        mask_np = qimage_to_numpy_gray(self.mask_image).astype(np.float32) / 255.0

        # Маскуємо спектр
        F_mod = self.dft_complex * mask_np

        # Для DFT-вікна показуємо лог-модуль
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

        # Відновлюємо зображення через зворотнє перетворення
        img_recon = np.fft.ifft2(np.fft.ifftshift(F_mod))
        img_recon = np.real(img_recon)

        # Нормуємо до [0, 255]
        img_recon -= img_recon.min()
        if img_recon.max() > 0:
            img_recon = img_recon / img_recon.max()
        img_recon_disp = (img_recon * 255).astype(np.uint8)

        img_qimage = numpy_to_qimage_gray(img_recon_disp)
        img_pixmap = QPixmap.fromImage(img_qimage)

        if self.img_pixmap_item is None:
            self.img_pixmap_item = QGraphicsPixmapItem(img_pixmap)
            self.img_scene.addItem(self.img_pixmap_item)
        else:
            self.img_pixmap_item.setPixmap(img_pixmap)

    # ---------- Малювання по масці ----------

    def begin_stroke(self):
        """
        Початок "штриха" по масці – зберігаємо поточну маску в undo_stack.
        """
        if self.mask_image is None:
            return
        self.undo_stack.append(self.mask_image.copy())
        self.redo_stack.clear()

    def get_pen_color(self):
        """
        Колір інструмента:
        - Eraser → білий (255)
        - інші → інтенсивність із повзунка
        """
        if self.current_tool == "eraser":
            intensity = 255
        else:
            intensity = self.brush_intensity
        return QColor(intensity, intensity, intensity)

    def paint_segment(self, p1: QPointF, p2: QPointF):
        """
        Малюємо відрізок (для олівця/ластика) між двома точками.
        """
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
        """
        Малюємо пряму лінію (Line tool).
        """
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
        """
        Малюємо коло.
        - circle: заповнене коло
        - inv_circle: ВСЯ маска затемнена, всередині кола — біле вікно
        """
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
            # Звичайне заповнене коло
            pen_color = self.get_pen_color()
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(pen_color, Qt.SolidPattern))
            painter.drawEllipse(left, top, w, h)
        else:
            # VARIANT A:
            # 1) Заливаємо ВСЮ маску темним кольором (придушуємо всю частоту)
            dark_color = self.get_pen_color()
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(dark_color, Qt.SolidPattern))
            painter.drawRect(0, 0, self.mask_image.width(), self.mask_image.height())

            # 2) Колом всередині робимо біле "вікно" (там спектр не змінюється)
            painter.setBrush(QBrush(QColor(255, 255, 255), Qt.SolidPattern))
            painter.drawEllipse(left, top, w, h)

        painter.end()

    # ---------- Undo / Redo / Reset ----------

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
        """
        Скидання всіх змін: маска = 255 (повністю біла, спектр без змін).
        """
        if not self.image_loaded:
            return
        h, w = self.original_image.shape
        mask_np = np.full((h, w), 255, dtype=np.uint8)
        self.mask_image = numpy_to_qimage_gray(mask_np)
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.update_from_mask()

    # ---------- Clipboard: Copy / Paste ----------

    def paste_image(self):
        """
        Ctrl+V – вставити зображення з буфера в original.
        """
        clipboard = QApplication.clipboard()
        img = clipboard.image()

        if img.isNull():
            return

        arr = qimage_to_numpy_gray(img)
        self.load_new_image(arr)

    def copy_image(self):
        """
        Ctrl+C – скопіювати відновлене зображення в буфер.
        """
        if self.img_pixmap_item is None:
            return

        pixmap = self.img_pixmap_item.pixmap()
        QApplication.clipboard().setPixmap(pixmap)

    # ---------- Drag & Drop ----------

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if not urls:
            return

        path = urls[0].toLocalFile()
        if not path:
            return

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return

        self.load_new_image(img)
        event.acceptProposedAction()


# ===== Запуск програми =====

def main():
    app = QApplication(sys.argv)
    win = FourierEditor()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
