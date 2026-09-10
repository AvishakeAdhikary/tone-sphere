"""
Node-graph routing view.

Replaces a hand-rolled pan/zoom implementation on a `tk.Canvas`. `QGraphicsScene` provides
the scene transform, hit testing, z-ordering and item picking that the previous version
approximated with manual coordinate arithmetic — which is why its cables were straight
lines and its hit testing was a bounding-box guess.

The interaction model is the one every patching interface uses, and it was worth keeping
from the original: drag from an output port to an input port to connect, right-click a
cable to remove it, drag nodes to arrange, wheel to zoom.
"""


from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QPainter,
    QPainterPath,
    QPen,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsObject,
    QGraphicsScene,
    QGraphicsView,
    QMenu,
    QStyleOptionGraphicsItem,
    QWidget,
)

from tonesphere.i18n import tr
from tonesphere.ui.theme import Colors, Spacing, Type, format_db

NODE_WIDTH = 172
NODE_HEIGHT = 56
PORT_RADIUS = 6
GRID = 24


class PortItem(QGraphicsObject):
    """
    A connection point. Inputs on the left, outputs on the right.

    Given a generous hit area on purpose: a 6 px circle is accurate to draw and miserable
    to hit, so the clickable region is roughly twice the visual one.
    """

    def __init__(self, node: "NodeItem", is_output: bool):
        super().__init__(node)

        self.node = node
        self.is_output = is_output
        self._hovered = False

        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setToolTip(tr('port.output_tooltip') if is_output else tr('port.input_tooltip'))

    def boundingRect(self) -> QRectF:
        r = PORT_RADIUS * 2
        return QRectF(-r, -r, r * 2, r * 2)

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem,
              widget: QWidget | None = None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        radius = PORT_RADIUS + (2 if self._hovered else 0)
        painter.setPen(QPen(Colors.BORDER_STRONG, 1))
        painter.setBrush(Colors.ACCENT if self._hovered else Colors.BG_ACTIVE)
        painter.drawEllipse(QPointF(0, 0), radius, radius)

    def hoverEnterEvent(self, event):
        self._hovered = True
        self.update()

    def hoverLeaveEvent(self, event):
        self._hovered = False
        self.update()

    def mousePressEvent(self, event):
        if self.is_output and event.button() == Qt.MouseButton.LeftButton:
            self.node.scene().begin_cable(self)
            event.accept()
        else:
            event.ignore()


class NodeItem(QGraphicsObject):
    """
    A device or bus in the graph.

    Carries its own level meter so signal flow is visible at a glance: you can see which
    part of the patch has audio in it without reading the mixer.
    """

    moved = Signal()

    def __init__(self, node_id: int, name: str, subtitle: str,
                 can_input: bool, can_output: bool, is_bus: bool = False):
        super().__init__()

        self.node_id = node_id
        self.name = name
        self.subtitle = subtitle
        self.is_bus = is_bus
        self.failed: str | None = None

        self._level = 0.0
        self._cables: list[CableItem] = []

        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

        self.input_port: PortItem | None = None
        self.output_port: PortItem | None = None

        if can_input:
            self.input_port = PortItem(self, is_output=False)
            self.input_port.setPos(0, NODE_HEIGHT / 2)
        if can_output:
            self.output_port = PortItem(self, is_output=True)
            self.output_port.setPos(NODE_WIDTH, NODE_HEIGHT / 2)

        self.setToolTip(tr('node.tooltip', name=name, subtitle=subtitle))

    def boundingRect(self) -> QRectF:
        return QRectF(-PORT_RADIUS * 2, 0,
                      NODE_WIDTH + PORT_RADIUS * 4, NODE_HEIGHT)

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem,
              widget: QWidget | None = None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        body = QRectF(0, 0, NODE_WIDTH, NODE_HEIGHT)

        if self.failed:
            border, width = Colors.ERROR, 2
        elif self.isSelected():
            border, width = Colors.ACCENT, 2
        else:
            border, width = Colors.BORDER_STRONG, 1

        painter.setPen(QPen(border, width))
        painter.setBrush(Colors.BG_PANEL if not self.is_bus else Colors.BG_RAISED)
        painter.drawRoundedRect(body, Spacing.RADIUS, Spacing.RADIUS)

        # A bus is visually distinct from hardware: it is in-process only, and conflating
        # the two is exactly the confusion the old UI created.
        if self.is_bus:
            painter.setPen(QPen(Colors.ACCENT_DIM, 2))
            painter.drawLine(QPointF(1, 4), QPointF(1, NODE_HEIGHT - 4))

        painter.setPen(Colors.TEXT if not self.failed else Colors.ERROR)
        painter.setFont(Type.font(Type.SMALL, weight=600))
        painter.drawText(
            QRectF(Spacing.MD, Spacing.SM, NODE_WIDTH - Spacing.XL, 16),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            painter.fontMetrics().elidedText(
                self.name, Qt.TextElideMode.ElideRight, NODE_WIDTH - Spacing.XL),
        )

        painter.setPen(Colors.ERROR if self.failed else Colors.TEXT_DIM)
        painter.setFont(Type.font(Type.TINY))
        subtitle = self.failed if self.failed else self.subtitle
        painter.drawText(
            QRectF(Spacing.MD, 20, NODE_WIDTH - Spacing.XL, 14),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            painter.fontMetrics().elidedText(
                subtitle, Qt.TextElideMode.ElideRight, NODE_WIDTH - Spacing.XL),
        )

        self._paint_level(painter)

    def _paint_level(self, painter: QPainter):
        """A slim level strip along the bottom edge: signal presence at a glance."""
        track = QRectF(Spacing.MD, NODE_HEIGHT - 12, NODE_WIDTH - Spacing.XL, 4)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(Colors.BG_SUNKEN)
        painter.drawRoundedRect(track, 2, 2)

        if self._level > 0.001:
            filled = QRectF(track)
            filled.setWidth(track.width() * min(self._level, 1.0))
            painter.setBrush(Colors.METER_LOW if self._level < 0.7 else Colors.METER_HIGH)
            painter.drawRoundedRect(filled, 2, 2)

    def set_level(self, level: float):
        if abs(level - self._level) > 0.005:
            self._level = level
            self.update()

    def set_failed(self, reason: str | None):
        self.failed = reason
        self.setToolTip(
            tr('node.failed_tooltip', name=self.name, reason=reason) if reason
            else tr('node.tooltip', name=self.name, subtitle=self.subtitle)
        )
        self.update()

    def register_cable(self, cable: "CableItem"):
        self._cables.append(cable)

    def unregister_cable(self, cable: "CableItem"):
        if cable in self._cables:
            self._cables.remove(cable)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            for cable in self._cables:
                cable.refresh()
        return super().itemChange(change, value)

    def input_scene_pos(self) -> QPointF:
        return self.mapToScene(QPointF(0, NODE_HEIGHT / 2))

    def output_scene_pos(self) -> QPointF:
        return self.mapToScene(QPointF(NODE_WIDTH, NODE_HEIGHT / 2))


class CableItem(QGraphicsObject):
    """
    A patch cable.

    Drawn as a bezier because a straight line between two nodes crosses whatever is between
    them and becomes unreadable the moment there is more than one. The horizontal control
    points make cables leave and enter ports at right angles, so parallel runs stay
    visually separate.
    """

    def __init__(self, source: NodeItem, dest: NodeItem,
                 gain_db: float = 0.0, muted: bool = False):
        super().__init__()

        self.source = source
        self.dest = dest
        self.gain_db = gain_db
        self.muted = muted
        self.dead = False

        self._path = QPainterPath()
        self._hovered = False

        # Behind nodes, so a cable never obscures the thing it connects.
        self.setZValue(-1)
        self.setAcceptHoverEvents(True)

        source.register_cable(self)
        dest.register_cable(self)
        self.refresh()

    @property
    def key(self) -> tuple[int, int]:
        return (self.source.node_id, self.dest.node_id)

    def refresh(self):
        self.prepareGeometryChange()

        start = self.source.output_scene_pos()
        end = self.dest.input_scene_pos()

        # Control-point offset scales with distance so short cables stay tight and long
        # ones bow enough to be followed by eye.
        offset = max(40.0, min(abs(end.x() - start.x()) * 0.55, 160.0))

        path = QPainterPath(start)
        path.cubicTo(QPointF(start.x() + offset, start.y()),
                     QPointF(end.x() - offset, end.y()),
                     end)
        self._path = path
        self.update()

    def boundingRect(self) -> QRectF:
        return self._path.boundingRect().adjusted(-6, -6, 6, 6)

    def shape(self) -> QPainterPath:
        """A thick stroke for hit testing: a 2 px curve is impossible to click otherwise."""
        from PySide6.QtGui import QPainterPathStroker

        stroker = QPainterPathStroker()
        stroker.setWidth(12)
        return stroker.createStroke(self._path)

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem,
              widget: QWidget | None = None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self.dead:
            colour, width, style = Colors.CABLE_DEAD, 2, Qt.PenStyle.DashLine
        elif self.muted:
            colour, width, style = Colors.CABLE_MUTED, 2, Qt.PenStyle.DashLine
        elif self._hovered:
            colour, width, style = Colors.CABLE_ACTIVE, 3, Qt.PenStyle.SolidLine
        else:
            colour, width, style = Colors.CABLE, 2, Qt.PenStyle.SolidLine

        painter.setPen(QPen(colour, width, style, Qt.PenCapStyle.RoundCap))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(self._path)

        if self._hovered or abs(self.gain_db) > 0.05:
            self._paint_label(painter)

    def _paint_label(self, painter: QPainter):
        midpoint = self._path.pointAtPercent(0.5)
        text = tr('cable.muted') if self.muted else format_db(self.gain_db)

        painter.setFont(Type.numeric(Type.TINY))
        metrics = painter.fontMetrics()
        width = metrics.horizontalAdvance(text) + Spacing.MD
        rect = QRectF(midpoint.x() - width / 2, midpoint.y() - 8, width, 15)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(Colors.BG_BASE)
        painter.drawRoundedRect(rect, 3, 3)

        painter.setPen(Colors.TEXT_MUTED if not self.muted else Colors.TEXT_DIM)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    def hoverEnterEvent(self, event):
        self._hovered = True
        self.update()

    def hoverLeaveEvent(self, event):
        self._hovered = False
        self.update()

    def detach(self):
        self.source.unregister_cable(self)
        self.dest.unregister_cable(self)


class RoutingScene(QGraphicsScene):
    """The scene: owns nodes and cables, and runs the drag-to-connect interaction."""

    connect_requested = Signal(int, int)      # source_id, dest_id
    disconnect_requested = Signal(int, int)
    gain_requested = Signal(int, int, float)
    mute_requested = Signal(int, int, bool)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setBackgroundBrush(Colors.BG_SUNKEN)

        self.nodes: dict[int, NodeItem] = {}
        self.cables: dict[tuple[int, int], CableItem] = {}

        self._pending_source: PortItem | None = None
        self._pending_path: QGraphicsItem | None = None

    # --- Content ---

    def add_node(self, node_id: int, name: str, subtitle: str,
                 can_input: bool, can_output: bool, is_bus: bool,
                 position: QPointF) -> NodeItem:
        node = NodeItem(node_id, name, subtitle, can_input, can_output, is_bus)
        node.setPos(position)
        self.addItem(node)
        self.nodes[node_id] = node
        return node

    def add_cable(self, source_id: int, dest_id: int,
                  gain_db: float = 0.0, muted: bool = False) -> CableItem | None:
        source = self.nodes.get(source_id)
        dest = self.nodes.get(dest_id)
        if source is None or dest is None:
            return None

        existing = self.cables.get((source_id, dest_id))
        if existing is not None:
            existing.gain_db = gain_db
            existing.muted = muted
            existing.update()
            return existing

        cable = CableItem(source, dest, gain_db, muted)
        self.addItem(cable)
        self.cables[(source_id, dest_id)] = cable
        return cable

    def remove_cable(self, source_id: int, dest_id: int):
        cable = self.cables.pop((source_id, dest_id), None)
        if cable is not None:
            cable.detach()
            self.removeItem(cable)

    def clear_content(self):
        for cable in list(self.cables.values()):
            cable.detach()
            self.removeItem(cable)
        self.cables.clear()

        for node in list(self.nodes.values()):
            self.removeItem(node)
        self.nodes.clear()

    # --- Patching ---

    def begin_cable(self, port: PortItem):
        self._pending_source = port
        self._pending_path = self.addPath(
            QPainterPath(),
            QPen(Colors.ACCENT, 2, Qt.PenStyle.DashLine, Qt.PenCapStyle.RoundCap),
        )
        self._pending_path.setZValue(10)

    def mouseMoveEvent(self, event):
        if self._pending_source is not None and self._pending_path is not None:
            start = self._pending_source.node.output_scene_pos()
            end = event.scenePos()

            offset = max(40.0, min(abs(end.x() - start.x()) * 0.55, 160.0))
            path = QPainterPath(start)
            path.cubicTo(QPointF(start.x() + offset, start.y()),
                         QPointF(end.x() - offset, end.y()), end)
            self._pending_path.setPath(path)

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._pending_source is not None:
            self._finish_cable(event.scenePos())
        super().mouseReleaseEvent(event)

    def _finish_cable(self, scene_pos: QPointF):
        source_node = self._pending_source.node

        if self._pending_path is not None:
            self.removeItem(self._pending_path)
            self._pending_path = None
        self._pending_source = None

        # Find a node under the drop point that can accept an input.
        for item in self.items(scene_pos):
            node = item if isinstance(item, NodeItem) else item.parentItem()
            if isinstance(node, NodeItem) and node.input_port is not None:
                if node.node_id != source_node.node_id:
                    self.connect_requested.emit(source_node.node_id, node.node_id)
                return

    def contextMenuEvent(self, event):
        item = self.itemAt(event.scenePos(), self.views()[0].transform())
        cable = item if isinstance(item, CableItem) else None

        if cable is None:
            super().contextMenuEvent(event)
            return

        menu = QMenu()
        source_id, dest_id = cable.key

        toggle = menu.addAction(tr('cable.menu.unmute') if cable.muted else tr('cable.menu.mute'))
        menu.addSeparator()
        unity = menu.addAction(tr('cable.menu.unity'))
        down6 = menu.addAction(tr('cable.menu.minus_6'))
        down12 = menu.addAction(tr('cable.menu.minus_12'))
        menu.addSeparator()
        remove = menu.addAction(tr('cable.menu.disconnect'))

        chosen = menu.exec(event.screenPos())

        if chosen == remove:
            self.disconnect_requested.emit(source_id, dest_id)
        elif chosen == toggle:
            self.mute_requested.emit(source_id, dest_id, not cable.muted)
        elif chosen == unity:
            self.gain_requested.emit(source_id, dest_id, 0.0)
        elif chosen == down6:
            self.gain_requested.emit(source_id, dest_id, -6.0)
        elif chosen == down12:
            self.gain_requested.emit(source_id, dest_id, -12.0)

    def drawBackground(self, painter: QPainter, rect: QRectF):
        """A dot grid, faint enough to give a sense of position without competing."""
        super().drawBackground(painter, rect)

        painter.setPen(QPen(Colors.alpha(Colors.BORDER, 0.7), 1))

        left = int(rect.left()) - (int(rect.left()) % GRID)
        top = int(rect.top()) - (int(rect.top()) % GRID)

        points = [
            QPointF(x, y)
            for x in range(left, int(rect.right()), GRID)
            for y in range(top, int(rect.bottom()), GRID)
        ]
        if len(points) < 8000:   # skip when zoomed far out; the dots would be a wash
            painter.drawPoints(points)


class RoutingView(QGraphicsView):
    """
    The viewport: pan, zoom and fit.

    Zoom is centred on the cursor, which is what makes navigating a large patch feel
    direct rather than like operating a machine.
    """

    MIN_SCALE = 0.25
    MAX_SCALE = 2.5

    def __init__(self, scene: RoutingScene, parent: QWidget | None = None):
        super().__init__(scene, parent)

        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.SmartViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self._scale = 1.0

    def wheelEvent(self, event: QWheelEvent):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        target = self._scale * factor

        if not (self.MIN_SCALE <= target <= self.MAX_SCALE):
            return

        self._scale = target
        self.scale(factor, factor)

    def mousePressEvent(self, event):
        # Middle-drag pans, the convention in every node editor.
        if event.button() == Qt.MouseButton.MiddleButton:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            forwarded = type(event)(
                event.type(), event.position(), event.globalPosition(),
                Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, event.modifiers(),
            )
            super().mousePressEvent(forwarded)
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            super().mouseReleaseEvent(event)
            self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
            return
        super().mouseReleaseEvent(event)

    def fit_content(self):
        """
        Frame the patch.

        Capped at 1:1 so a small patch is shown at natural size rather than blown up to
        fill the viewport, which would make two nodes look like a poster.
        """
        items_rect = self.scene().itemsBoundingRect()
        if items_rect.isEmpty():
            return

        self.fitInView(items_rect.adjusted(-50, -50, 50, 50),
                       Qt.AspectRatioMode.KeepAspectRatio)

        scale = self.transform().m11()
        if scale > 1.0:
            self.resetTransform()
            self.centerOn(items_rect.center())
            scale = 1.0

        self._scale = scale

    def reset_zoom(self):
        self.resetTransform()
        self._scale = 1.0
