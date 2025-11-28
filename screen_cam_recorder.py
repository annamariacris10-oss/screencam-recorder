"""
screen_recorder_no_ffmpeg.py

- Records screen-> MP4 (OpenCV) and microphone-> WAV (sounddevice+soundfile) simultaneously
- Live GUI with realtime webcam preview + draggable floating webcam (overlay position used when writing video)
- Camera Pause button: freezes webcam (frozen frame used as overlay while recording)
- NO ffmpeg used. Audio and video files are saved separately.
"""

import os
import time
import threading
import tempfile
from datetime import timedelta
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
from mss import mss
from PIL import Image, ImageTk
import sounddevice as sd
import soundfile as sf

# -------------------------
# Config / utils
# -------------------------
TEMP_DIR = tempfile.gettempdir()
DEFAULT_FPS = 15
DEFAULT_WEBCAM_SIZE = (320, 240)


class AudioRecorder:
    """Record microphone audio to a WAV file using sounddevice & soundfile."""
    def __init__(self, filename, samplerate=44100, channels=1, subtype='PCM_16'):
        self.filename = filename
        self.sr = samplerate
        self.channels = channels
        self.subtype = subtype
        self._file = None
        self._stream = None

    def _callback(self, indata, frames, time_info, status):
        if status:
            # non-fatal status messages (optional)
            pass
        if self._file is not None:
            self._file.write(indata.copy())

    def start(self):
        self._file = sf.SoundFile(self.filename, mode='w', samplerate=self.sr, channels=self.channels, subtype=self.subtype)
        self._stream = sd.InputStream(samplerate=self.sr, channels=self.channels, callback=self._callback)
        self._stream.start()

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._file is not None:
            self._file.close()
            self._file = None


class ScreenRecorder:
    """Record screen to a video file and overlay webcam frames (or frozen webcam frame)"""
    def __init__(self, fps=DEFAULT_FPS, webcam_index=0):
        self.fps = fps
        self.webcam_index = webcam_index
        self._running = False
        self._thread = None

        # temp filenames assigned when start() is called
        self.video_path = None
        self.audio_path = None

        # control/state passed in from GUI
        self.get_overlay_params = None  # callable that returns dict: {enabled, size, shape, opacity, floating_pos_getter, camera_pause_flag, frozen_frame_getter}

    def start(self, video_path, audio_path):
        if self._running:
            return
        self.video_path = video_path
        self.audio_path = audio_path
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def _rounded_mask(self, w, h, rad):
        mask = np.zeros((h, w), dtype=np.uint8)
        # create rectangle then round corners by eroding
        mask[:] = 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rad, rad))
        mask = cv2.erode(mask, kernel)
        return mask

    def _circle_mask(self, w, h):
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (w // 2, h // 2), min(w, h) // 2, 255, -1)
        return mask

    def _run(self):
        try:
            sct = mss()
            monitor = sct.monitors[1]  # primary monitor
            mon_left, mon_top = monitor['left'], monitor['top']
            width, height = monitor['width'], monitor['height']

            # Video writer (mp4)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(self.video_path, fourcc, self.fps, (width, height))

            # Webcam capture
            cap = cv2.VideoCapture(self.webcam_index)
            if not cap.isOpened():
                cap = None
                print("Warning: webcam not available. Overlay disabled.")

            # Start audio recording concurrently
            audio_rec = AudioRecorder(self.audio_path)
            audio_rec.start()

            frame_time = 1.0 / max(1, self.fps)
            next_time = time.time()

            while self._running:
                sct_img = sct.grab(monitor)
                frame = np.array(sct_img)  # BGRA
                # convert to BGR if needed
                if frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                # overlay webcam based on GUI-provided params
                params = None
                if self.get_overlay_params:
                    try:
                        params = self.get_overlay_params()
                    except Exception as e:
                        params = None

                if params and params.get('enabled', True) and cap is not None:
                    # fetch cam frame: either live or frozen from GUI
                    if params.get('camera_paused', False):
                        cam = params.get('frozen_frame')  # may be None if never captured
                        if cam is None:
                            # fallback to reading from camera once
                            ret, cam = cap.read()
                            if not ret:
                                cam = None
                    else:
                        ret, cam = cap.read()
                        if not ret:
                            cam = None

                    if cam is not None:
                        w_cam, h_cam = params.get('size', (320, 240))
                        cam_small = cv2.resize(cam, (w_cam, h_cam))
                        # position from floating window (screen coords)
                        getter = params.get('floating_getter')
                        if getter:
                            wx, wy = getter()
                            x = int(wx - mon_left)
                            y = int(wy - mon_top)
                        else:
                            x = width - w_cam - 20
                            y = height - h_cam - 20
                        x = max(0, min(x, width - w_cam))
                        y = max(0, min(y, height - h_cam))

                        shape = params.get('shape', 'square')
                        opacity = float(params.get('opacity', 1.0))

                        if shape == 'square':
                            if opacity >= 0.99:
                                frame[y:y+h_cam, x:x+w_cam] = cam_small
                            else:
                                alpha = opacity
                                bg = frame[y:y+h_cam, x:x+w_cam].astype(np.float32)
                                fg = cam_small.astype(np.float32)
                                blended = (fg * alpha + bg * (1 - alpha)).astype(np.uint8)
                                frame[y:y+h_cam, x:x+w_cam] = blended
                        elif shape == 'rounded':
                            mask = self._rounded_mask(w_cam, h_cam, rad=max(8, min(w_cam, h_cam)//8))
                            mask3 = cv2.merge([mask, mask, mask])
                            fg = cv2.bitwise_and(cam_small, mask3)
                            bg = cv2.bitwise_and(frame[y:y+h_cam, x:x+w_cam], cv2.bitwise_not(mask3))
                            frame[y:y+h_cam, x:x+w_cam] = cv2.add(bg, fg)
                        elif shape == 'circle':
                            mask = self._circle_mask(w_cam, h_cam)
                            mask3 = cv2.merge([mask, mask, mask])
                            fg = cv2.bitwise_and(cam_small, mask3)
                            bg = cv2.bitwise_and(frame[y:y+h_cam, x:x+w_cam], cv2.bitwise_not(mask3))
                            frame[y:y+h_cam, x:x+w_cam] = cv2.add(bg, fg)

                writer.write(frame)

                # timing
                next_time += frame_time
                sleep_time = next_time - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    next_time = time.time()

            # cleanup
            writer.release()
            if cap is not None:
                cap.release()
            audio_rec.stop()

        except Exception as e:
            print("Recorder error:", e)
            try:
                audio_rec.stop()
            except Exception:
                pass


# -------------------------
# GUI
# -------------------------
class App:
    def __init__(self, root):
        self.root = root
        root.title("Screen Recorder (No ffmpeg) — Audio saved to WAV")
        root.geometry("560x560")
        root.resizable(False, False)

        # Backend
        self.rec = ScreenRecorder()
        self.rec.get_overlay_params = self._overlay_params_getter

        # Camera for preview
        self.webcam_index = 0
        self.cap = cv2.VideoCapture(self.webcam_index)
        self.last_cam_frame = None  # frozen frame when camera is paused

        # State
        self.is_recording = False
        self.camera_paused = False
        self.start_time = None
        self.timer_job = None

        # UI vars
        self.fps_var = tk.IntVar(value=DEFAULT_FPS)
        self.wcam_w = tk.IntVar(value=DEFAULT_WEBCAM_SIZE[0])
        self.wcam_h = tk.IntVar(value=DEFAULT_WEBCAM_SIZE[1])
        self.shape_var = tk.StringVar(value="rounded")
        self.opacity_var = tk.DoubleVar(value=1.0)
        self.output_var = tk.StringVar(value=os.path.join(os.getcwd(), "output_noffmpeg.mp4"))

        self._build_ui()
        self._create_floating_cam()
        self._update_preview_loop()

    def _build_ui(self):
        pad = 8
        frm = ttk.Frame(self.root, padding=pad)
        frm.pack(fill="both", expand=True)

        ttk.Label(frm, text="Screen Recorder (no ffmpeg)", font=("Segoe UI", 16, "bold")).pack(pady=(2, 6))

        # preview
        preview_box = ttk.LabelFrame(frm, text="Live Camera Preview")
        preview_box.pack(fill="x", padx=pad, pady=(0, pad))
        self.preview_label = ttk.Label(preview_box)
        self.preview_label.pack(padx=8, pady=8)

        # controls
        ctrl_box = ttk.LabelFrame(frm, text="Settings")
        ctrl_box.pack(fill="x", padx=pad, pady=(0, pad))

        row = ttk.Frame(ctrl_box)
        row.pack(fill="x", pady=4, padx=8)
        ttk.Label(row, text="FPS:").pack(side="left")
        ttk.Spinbox(row, from_=5, to=60, textvariable=self.fps_var, width=5).pack(side="left", padx=6)
        ttk.Label(row, text="Webcam size:").pack(side="left", padx=(12,2))
        ttk.Entry(row, textvariable=self.wcam_w, width=6).pack(side="left")
        ttk.Label(row, text="x").pack(side="left")
        ttk.Entry(row, textvariable=self.wcam_h, width=6).pack(side="left")

        row2 = ttk.Frame(ctrl_box)
        row2.pack(fill="x", pady=4, padx=8)
        ttk.Label(row2, text="Overlay shape:").pack(side="left")
        ttk.OptionMenu(row2, self.shape_var, self.shape_var.get(), "square", "rounded", "circle").pack(side="left", padx=6)
        ttk.Label(row2, text="Opacity:").pack(side="left", padx=(12,2))
        ttk.Scale(row2, variable=self.opacity_var, from_=0.2, to=1.0, orient="horizontal").pack(side="left", padx=6, fill="x", expand=True)

        out_row = ttk.Frame(ctrl_box)
        out_row.pack(fill="x", pady=6, padx=8)
        ttk.Label(out_row, text="Output (video mp4):").pack(side="left")
        ttk.Entry(out_row, textvariable=self.output_var, width=36).pack(side="left", padx=6)
        ttk.Button(out_row, text="Browse", command=self._browse_out).pack(side="left")

        # buttons
        btns = ttk.Frame(frm)
        btns.pack(fill="x", pady=(8,4), padx=8)
        self.start_btn = ttk.Button(btns, text="Start Recording", command=self._start)
        self.start_btn.pack(side="left", padx=6)
        self.stop_btn = ttk.Button(btns, text="Stop Recording", command=self._stop, state="disabled")
        self.stop_btn.pack(side="left", padx=6)
        self.cam_pause_btn = ttk.Button(btns, text="Pause Camera", command=self._toggle_camera_pause)
        self.cam_pause_btn.pack(side="left", padx=6)

        # status/timer
        status = ttk.Frame(frm)
        status.pack(fill="x", padx=8, pady=(6,0))
        self.status_var = tk.StringVar(value="Ready")
        self.status_lbl = ttk.Label(status, textvariable=self.status_var, foreground="green")
        self.status_lbl.pack(side="left")
        self.timer_var = tk.StringVar(value="00:00")
        self.timer_lbl = ttk.Label(status, textvariable=self.timer_var, font=("Segoe UI", 12))
        self.timer_lbl.pack(side="right")

        ttk.Label(frm, text="Tip: drag the floating webcam window to move overlay position.", foreground="gray").pack(pady=(6,0))

    def _browse_out(self):
        file = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4","*.mp4"), ("All files","*.*")])
        if file:
            self.output_var.set(file)

    def _create_floating_cam(self):
        self.float_win = tk.Toplevel(self.root)
        self.float_win.title("Floating Webcam (drag me)")
        w, h = self.wcam_w.get(), self.wcam_h.get()
        self.float_win.geometry(f"{w}x{h}+120+120")
        self.float_win.attributes("-topmost", True)
        self.float_win.resizable(False, False)
        self.float_label = tk.Label(self.float_win, bg="black")
        self.float_label.pack(fill="both", expand=True)
        # drag
        self._drag = {"x":0,"y":0}
        self.float_label.bind("<ButtonPress-1>", self._on_float_press)
        self.float_label.bind("<B1-Motion>", self._on_float_move)
        self.float_win.protocol("WM_DELETE_WINDOW", self._on_float_close)

    def _on_float_press(self, ev):
        self._drag['x'] = ev.x
        self._drag['y'] = ev.y

    def _on_float_move(self, ev):
        dx = ev.x - self._drag['x']
        dy = ev.y - self._drag['y']
        geom = self.float_win.geometry().split('+')
        size = geom[0]
        try:
            curx = int(geom[1]); cury = int(geom[2])
        except:
            return
        newx = curx + dx; newy = cury + dy
        self.float_win.geometry(f"{size}+{newx}+{newy}")

    def _on_float_close(self):
        # hide instead of destroy
        self.float_win.withdraw()

    def _get_float_pos(self):
        try:
            return self.float_win.winfo_rootx(), self.float_win.winfo_rooty()
        except:
            return 100, 100

    def _overlay_params_getter(self):
        """Return overlay params for recorder"""
        return {
            'enabled': True,
            'size': (max(80, int(self.wcam_w.get())), max(60, int(self.wcam_h.get()))),
            'shape': self.shape_var.get(),
            'opacity': max(0.2, min(1.0, float(self.opacity_var.get()))),
            'floating_getter': self._get_float_pos,
            'camera_paused': self.camera_paused,
            'frozen_frame': self.last_cam_frame
        }

    def _update_preview_loop(self):
        # update preview and floating window image
        ret, frame = self.cap.read()
        if ret:
            self.last_cam_frame = frame.copy()
            # main preview
            disp_h = 220
            h, w = frame.shape[:2]
            scale = disp_h / h
            img = cv2.resize(frame, (int(w*scale), disp_h))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            im_pil = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(im_pil)
            self.preview_label.imgtk = imgtk
            self.preview_label.config(image=imgtk)

            # floating window render according to selected shape & size
            try:
                wcam = max(80, int(self.wcam_w.get()))
                hcam = max(60, int(self.wcam_h.get()))
                cam_small = cv2.resize(frame, (wcam, hcam))
                shape = self.shape_var.get()
                if shape == 'square':
                    disp = cam_small
                elif shape == 'rounded':
                    rad = max(8, min(wcam, hcam)//8)
                    mask = np.zeros((hcam, wcam), dtype=np.uint8)
                    mask[:] = 255
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rad, rad))
                    mask = cv2.erode(mask, kernel)
                    mask3 = cv2.merge([mask, mask, mask])
                    disp = cv2.bitwise_and(cam_small, mask3)
                else:  # circle
                    mask = np.zeros((hcam, wcam), dtype=np.uint8)
                    cv2.circle(mask, (wcam//2, hcam//2), min(wcam, hcam)//2, 255, -1)
                    mask3 = cv2.merge([mask, mask, mask])
                    disp = cv2.bitwise_and(cam_small, mask3)

                disp = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(disp)
                imgtk2 = ImageTk.PhotoImage(pil)
                self.float_label.imgtk = imgtk2
                self.float_label.config(image=imgtk2)

                # ensure float window size matches entries
                cur_geom = self.float_win.geometry()
                size = cur_geom.split('+')[0]
                if size != f"{wcam}x{hcam}":
                    try:
                        x = int(cur_geom.split('+')[1]); y = int(cur_geom.split('+')[2])
                    except:
                        x, y = 120, 120
                    self.float_win.geometry(f"{wcam}x{hcam}+{x}+{y}")

            except Exception as ex:
                print("Preview render error:", ex)
        else:
            # camera not available - show black
            pass

        # schedule next
        self.root.after(30, self._update_preview_loop)

    # -------------
    # Controls
    # -------------
    def _start(self):
        if self.is_recording:
            return
        video_out = self.output_var.get().strip()
        if not video_out:
            messagebox.showerror("Choose output", "Choose an output file for the video (.mp4).")
            return

        # ensure directory exists
        odir = os.path.dirname(video_out) or "."
        os.makedirs(odir, exist_ok=True)

        # prepare temp audio filename
        base = os.path.splitext(video_out)[0]
        audio_out = base + "_mic.wav"

        # configure backend
        self.rec.fps = max(5, int(self.fps_var.get()))
        self.rec.webcam_index = self.webcam_index

        # start recording threads
        self.rec.start(video_out, audio_out)

        self.is_recording = True
        self.start_time = time.time()
        self._update_timer()
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.status_var.set("● Recording")
        self.status_lbl.config(foreground='red')

    def _stop(self):
        if not self.is_recording:
            return
        self.rec.stop()
        self.is_recording = False
        if self.timer_job:
            self.root.after_cancel(self.timer_job)
            self.timer_job = None
        self.timer_var.set("00:00")
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.status_var.set("Recording stopped — video and audio saved separately")
        self.status_lbl.config(foreground='green')
        messagebox.showinfo("Saved", f"Video: {self.output_var.get()}\nAudio: {os.path.splitext(self.output_var.get())[0] + '_mic.wav'}")

    def _update_timer(self):
        if not self.is_recording:
            return
        elapsed = int(time.time() - self.start_time)
        self.timer_var.set(str(timedelta(seconds=elapsed))[2:] if elapsed >= 3600 else time.strftime("%M:%S", time.gmtime(elapsed)))
        self.timer_job = self.root.after(1000, self._update_timer)

    def _toggle_camera_pause(self):
        self.camera_paused = not self.camera_paused
        self.cam_pause_btn.config(text=("Resume Camera" if self.camera_paused else "Pause Camera"))

    def on_close(self):
        if self.is_recording:
            if not messagebox.askyesno("Quit", "Recording in progress. Stop and quit?"):
                return
            self.rec.stop()
        try:
            self.cap.release()
        except:
            pass
        try:
            self.float_win.destroy()
        except:
            pass
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
