"""Evaluacion inicial: mide la postura 'habitual' del usuario durante N segundos
y da un veredicto final, en vez de corregir cada frame.

Esos mismos segundos sirven de calibracion: los umbrales por defecto son
iguales para todos, pero cada cuerpo y cada camara son distintos, asi que al
terminar se proponen umbrales ajustados a esa persona.
"""
import time
from collections import Counter

from posture_metrics import DEFAULT_THRESHOLDS, PostureCalibrator, summarize_aspects

# Por debajo de esto el veredicto se apoya en muy poco tiempo y se avisa.
MIN_MEASURED_PERCENT = 70.0


class PostureAssessment:
    def __init__(self, duration=30.0, good_threshold=70.0, calibrate=True):
        self.duration = duration
        self._good_threshold = good_threshold
        self._calibrate = calibrate
        self._start_time = None
        self._done = False
        self._correct = 0
        self._incorrect = 0
        self._unmeasured = 0
        self._issues = Counter()
        self._contexts = Counter()
        self._aspect_totals = {}
        self._calibrator = PostureCalibrator()

    def start(self):
        self._start_time = time.time()
        self._done = False
        self._correct = 0
        self._incorrect = 0
        self._unmeasured = 0
        self._issues = Counter()
        self._contexts = Counter()
        self._aspect_totals = {}
        self._calibrator = PostureCalibrator()

    @property
    def active(self):
        """True mientras se esta recolectando (antes de finish())."""
        return self._start_time is not None and not self._done

    @property
    def elapsed(self):
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    @property
    def remaining(self):
        return max(0.0, self.duration - self.elapsed)

    @property
    def time_up(self):
        """True cuando ya se cumplio el tiempo y toca llamar a finish()."""
        return self.active and self.elapsed >= self.duration

    def record(self, is_correct, feedback_text, context=None, aspects=None, metrics=None):
        if not self.active:
            return
        if is_correct is None:
            # A 45 grados o muy lejos no se pudo medir: no es buena ni mala
            # postura, y contarla como mala bajaba el porcentaje sin motivo.
            self._unmeasured += 1
            return
        if is_correct:
            self._correct += 1
        else:
            self._incorrect += 1
            if feedback_text:
                self._issues[feedback_text] += 1
        if context and context != "no_determinado":
            self._contexts[context] += 1
        if aspects:
            for key, ok in aspects.items():
                total, good = self._aspect_totals.get(key, (0, 0))
                self._aspect_totals[key] = (total + 1, good + (1 if ok else 0))
        if metrics:
            self._calibrator.record(metrics)

    def finish(self, base_thresholds=None, anchor_thresholds=None):
        """Cierra la evaluacion y devuelve el veredicto. Solo debe llamarse una vez."""
        self._done = True
        total = self._correct + self._incorrect

        if total == 0:
            verdict = (
                "No pude medirte durante la evaluación. Ubícate de frente o de perfil a la cámara e intenta de nuevo."
                if self._unmeasured
                else "No pude verte bien durante la evaluación. Ubícate frente a la cámara e intenta de nuevo."
            )
            return {
                "samples": 0,
                "measured_percent": 0.0,
                "percent_correct": 0.0,
                "good": False,
                "dominant_issue": None,
                "dominant_context": None,
                "aspect_percents": {},
                "thresholds": None,
                "calibration_note": "Sin datos para calibrar.",
                "verdict": verdict,
            }

        percent_correct = (self._correct / total) * 100
        good = percent_correct >= self._good_threshold
        dominant_issue = self._issues.most_common(1)[0][0] if self._issues else None
        dominant_context = self._contexts.most_common(1)[0][0] if self._contexts else None
        context_phrase = {
            "sentado": " (mayormente sentado)",
            "de_pie": " (mayormente de pie)",
        }.get(dominant_context, "")
        aspect_percents = {
            key: (good_count / aspect_total * 100)
            for key, (aspect_total, good_count) in self._aspect_totals.items()
            if aspect_total > 0
        }

        # --- Calibracion personal ---
        thresholds = None
        calibration_note = "Calibración desactivada."
        if self._calibrate:
            base = base_thresholds or DEFAULT_THRESHOLDS
            suggested, calibration_note = self._calibrator.suggest(
                base=base, percent_correct=percent_correct, anchor=anchor_thresholds
            )
            # Solo se devuelve si realmente cambio algo respecto a la base.
            thresholds = suggested if suggested != base else None

        if good:
            verdict = (
                f"Evaluación completa. Tu postura habitual es buena, estuvo correcta el "
                f"{percent_correct:.0f} por ciento del tiempo{context_phrase}. El detalle está en pantalla."
            )
        else:
            verdict = (
                f"Evaluación completa. Tu postura necesita mejorar, estuvo correcta el "
                f"{percent_correct:.0f} por ciento del tiempo{context_phrase}. El detalle está en pantalla."
            )

        return {
            "samples": total,
            "measured_percent": total / (total + self._unmeasured) * 100,
            "percent_correct": percent_correct,
            "good": good,
            "dominant_issue": dominant_issue,
            "dominant_context": dominant_context,
            "aspect_percents": aspect_percents,
            "thresholds": thresholds,
            "calibration_note": calibration_note,
            "verdict": verdict,
        }


def verdict_lines(result):
    """Las lineas que se muestran en pantalla al terminar la evaluacion."""
    if not result["samples"]:
        return [result["verdict"]]
    lines = [f"{result['percent_correct']:.0f}% correcta en la evaluación"]
    lines += summarize_aspects(result["aspect_percents"])
    if result["measured_percent"] < MIN_MEASURED_PERCENT:
        lines.append(
            f"Solo pude medirte el {result['measured_percent']:.0f}% del tiempo: "
            "ubícate de frente o de perfil."
        )
    if result.get("calibration_note"):
        lines.append(result["calibration_note"])
    return lines
