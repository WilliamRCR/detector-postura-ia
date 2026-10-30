"""Audio del reto: voz del entrenador, efectos y musica, cada uno por su canal para
que suenen a la vez. Reproduce archivos de audio/, asi que funciona sin internet.

Si no hay salida de audio o falta un archivo, el juego sigue en silencio: un
problema con la bocina no puede detener el stand.
"""
import pathlib
import random

import pygame

import reto_frases

CARPETA = pathlib.Path(__file__).resolve().parent / "audio"
CANAL_VOZ = 0
VOLUMEN_MUSICA = 0.32


class Audio:
    def __init__(self, carpeta=CARPETA):
        self.carpeta = pathlib.Path(carpeta)
        self.disponible = False
        self.entrenador = None
        self._voces = {}
        self._efectos = {}
        self._ultima = {}
        self._pendiente = None
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            pygame.mixer.set_num_channels(10)
            pygame.mixer.set_reserved(1)
            self._voz = pygame.mixer.Channel(CANAL_VOZ)
            self.disponible = True
        except pygame.error as exc:
            print(f"Sin audio ({exc}); el juego sigue en silencio.")

    def _cargar(self, ruta):
        try:
            return pygame.mixer.Sound(str(ruta))
        except (pygame.error, FileNotFoundError) as exc:
            print(f"No pude cargar {ruta.name} ({exc}).")
            return None

    def elegir_entrenador(self, nombre):
        self.entrenador = nombre
        if not self.disponible or nombre in self._voces:
            return
        self._voces[nombre] = {
            archivo: self._cargar(self.carpeta / nombre / f"{archivo}.ogg") for archivo in reto_frases.textos(nombre)
        }

    def _frase(self, clave):
        """Una version de la frase, sin repetir la que acaba de sonar."""
        opciones = reto_frases.variantes(self.entrenador, clave)
        if len(opciones) > 1:
            opciones = [o for o in opciones if o != self._ultima.get(clave)]
        archivo = random.choice(opciones)
        self._ultima[clave] = archivo
        return self._voces.get(self.entrenador, {}).get(archivo)

    def decir(self, clave, ahora_mismo=False):
        """Sin `ahora_mismo`, si ya esta hablando queda solo la ultima frase en
        espera: las repeticiones van mas rapido que la voz."""
        if not self.disponible:
            return
        sonido = self._frase(clave)
        if sonido is None:
            return
        if ahora_mismo or not self._voz.get_busy():
            self._pendiente = None
            self._voz.play(sonido)
        else:
            self._pendiente = clave

    def actualizar(self):
        if self.disponible and self._pendiente and not self._voz.get_busy():
            clave, self._pendiente = self._pendiente, None
            self.decir(clave)

    def efecto(self, nombre):
        if not self.disponible:
            return
        if nombre not in self._efectos:
            self._efectos[nombre] = self._cargar(self.carpeta / "sonidos" / f"{nombre}.ogg")
        sonido = self._efectos[nombre]
        if sonido is not None:
            sonido.play()

    def musica(self, nombre):
        if not self.disponible:
            return
        try:
            pygame.mixer.music.load(str(self.carpeta / "sonidos" / f"{nombre}.ogg"))
            pygame.mixer.music.set_volume(VOLUMEN_MUSICA)
            pygame.mixer.music.play(-1, fade_ms=300)
        except pygame.error as exc:
            print(f"No pude poner la musica ({exc}).")

    def parar_musica(self):
        if self.disponible:
            pygame.mixer.music.fadeout(400)

    def silenciar_voz(self):
        if self.disponible:
            self._pendiente = None
            self._voz.stop()
