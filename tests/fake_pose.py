"""Poses falsas de MediaPipe para poder testear sin camara ni modelo.

Genera los 33 landmarks con la misma forma que devuelve MediaPipe, de modo que
los tests ejerciten el codigo real de core/posture_metrics.
"""
import math

import core


class FakeLandmark:
    __slots__ = ("x", "y", "z", "visibility")

    def __init__(self, x, y, z=0.0, visibility=1.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility

    def __repr__(self):
        return f"FakeLandmark({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"


def make_landmarks(
    shoulder_width=0.30,
    shoulder_y=0.40,
    hip_y=0.75,
    center_x=0.50,
    shoulder_tilt=0.0,
    head_forward=0.0,
    ear_gap=0.10,
    legs_visible=True,
    knee_angle=90.0,
    scale=1.0,
):
    """Construye una pose frontal parametrizada.

    - shoulder_tilt: diferencia de altura entre hombros (postura ladeada)
    - head_forward:  cuanto se adelanta la nariz en z (cabeza adelantada)
    - ear_gap:       distancia vertical oreja-hombro (cuello estirado o hundido)
    - knee_angle:    ~180 de pie, ~90 sentado
    - scale:         distancia a la camara. Escala TODO el cuerpo por igual,
                     asi que la misma postura de lejos solo cambia de tamano.
    """
    half = (shoulder_width * scale) / 2.0
    ear_gap = ear_gap * scale
    shoulder_tilt = shoulder_tilt * scale
    head_forward = head_forward * scale
    hip_y = shoulder_y + (hip_y - shoulder_y) * scale

    points = [FakeLandmark(center_x, shoulder_y, 0.0) for _ in range(core.NUM_LANDMARKS)]

    # Nariz
    points[core.NOSE] = FakeLandmark(center_x, shoulder_y - ear_gap - 0.05 * scale, -head_forward)

    # Orejas
    ear_y = shoulder_y - ear_gap
    points[core.EAR_L] = FakeLandmark(center_x - half * 0.35, ear_y, 0.0)
    points[core.EAR_R] = FakeLandmark(center_x + half * 0.35, ear_y, 0.0)

    # Hombros (el izquierdo sube/baja segun shoulder_tilt)
    points[core.SHOULDER_L] = FakeLandmark(center_x - half, shoulder_y - shoulder_tilt / 2, 0.0)
    points[core.SHOULDER_R] = FakeLandmark(center_x + half, shoulder_y + shoulder_tilt / 2, 0.0)

    # Caderas
    hip_half = half * 0.7
    points[core.HIP_L] = FakeLandmark(center_x - hip_half, hip_y, 0.0)
    points[core.HIP_R] = FakeLandmark(center_x + hip_half, hip_y, 0.0)

    # Piernas: rodillas y tobillos coherentes con knee_angle
    visibility = 1.0 if legs_visible else 0.1
    knee_y = hip_y + 0.12 * scale
    rad = math.radians(180.0 - knee_angle)
    ankle_dx = math.sin(rad) * 0.14 * scale
    ankle_dy = math.cos(rad) * 0.14 * scale
    for hip_i, knee_i, ankle_i, side in ((23, 25, 27, -1), (24, 26, 28, 1)):
        hip = points[hip_i]
        hip.visibility = visibility
        points[knee_i] = FakeLandmark(hip.x, knee_y, 0.0, visibility)
        points[ankle_i] = FakeLandmark(hip.x + side * ankle_dx, knee_y + ankle_dy, 0.0, visibility)

    return points


def make_profile_landmarks():
    """Pose de perfil: los hombros se superponen y el ancho colapsa."""
    return make_landmarks(shoulder_width=0.01)


def good_posture():
    """Pose frontal erguida: hombros nivelados, cabeza atras y cuello estirado."""
    return make_landmarks(shoulder_tilt=0.0, head_forward=0.0, ear_gap=0.12)


def bad_posture():
    """Pose encorvada: hombros ladeados, cabeza adelantada y cuello hundido."""
    return make_landmarks(shoulder_tilt=0.10, head_forward=0.25, ear_gap=0.03)
