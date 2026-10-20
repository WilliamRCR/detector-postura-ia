"""Evaluacion inicial: mide la postura 'habitual' del usuario durante N segundos
y da un veredicto final, en vez de corregir cada frame."""
import time
from collections import Counter


class PostureAssessment:
    def __init__(self, duration=30.0, good_threshold=70.0):
        self.duration = duration
        self._good_threshold = good_threshold
        self._start_time = None
        self._done = False
        self._correct = 0
        self._incorrect = 0
        self._issues = Counter()
        self._contexts = Counter()
        self._aspect_totals = {}

    def start(self):
        self._start_time = time.time()
        self._done = False
        self._correct = 0
        self._incorrect = 0
        self._issues = Counter()
        self._contexts = Counter()
        self._aspect_totals = {}

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

    def record(self, is_correct, feedback_text, context=None, aspects=None):
        if not self.active:
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

    def finish(self):
        """Cierra la evaluacion y devuelve el veredicto. Solo debe llamarse una vez."""
        self._done = True
        total = self._correct + self._incorrect

        if total == 0:
            return {
                "samples": 0,
                "percent_correct": 0.0,
                "good": False,
                "dominant_issue": None,
                "dominant_context": None,
                "aspect_percents": {},
                "verdict": "No pude verte bien durante la evaluacion. Ubicate frente a la camara e intenta de nuevo.",
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

        if good:
            verdict = (
                f"Evaluacion completa. Tu postura habitual es buena, estuvo correcta el "
                f"{percent_correct:.0f} por ciento del tiempo{context_phrase}. El detalle esta en pantalla."
            )
        else:
            verdict = (
                f"Evaluacion completa. Tu postura necesita mejorar, estuvo correcta el "
                f"{percent_correct:.0f} por ciento del tiempo{context_phrase}. El detalle esta en pantalla."
            )

        return {
            "samples": total,
            "percent_correct": percent_correct,
            "good": good,
            "dominant_issue": dominant_issue,
            "dominant_context": dominant_context,
            "aspect_percents": aspect_percents,
            "verdict": verdict,
        }
