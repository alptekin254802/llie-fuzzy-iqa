"""Responsive local-only capture UI; the worker is the sole owner of the camera.

Preview packets may be dropped. Scientific frames are read and saved serially by
capture.acquire, without an intervening preview read or a changed QC rule.
"""
import queue
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import c_exploratory as policy


class PreviewCamera:
    """Pass camera operations through unchanged; publish copies only for display."""
    def __init__(self, cap, pictures, events):
        self.cap, self.pictures, self.events = cap, pictures, events

    def __getattr__(self, name):
        return getattr(self.cap, name)

    def publish(self, im, metrics=None):
        packet = (im.copy(), metrics)
        try:
            self.pictures.get_nowait()
        except queue.Empty:
            pass
        self.pictures.put_nowait(packet)

    def read(self):
        ok, im = self.cap.read()
        if ok and im is not None:
            self.publish(im)
        return ok, im

    def progress(self, text):
        self.events.put(('progress', text))


class CameraWorker(threading.Thread):
    def __init__(self, acquire, args, api=policy):
        super().__init__(name='paired-exposure-camera', daemon=True)
        self.acquire, self.args, self.api = acquire, args, api
        self.commands = queue.Queue(maxsize=1)
        self.pictures = queue.Queue(maxsize=1)
        self.events = queue.Queue()
        self.closing = threading.Event()

    def run(self):
        cap = None
        try:
            self.events.put(('progress', 'Opening camera; verifying YUY2 and locked settings...'))
            cap = PreviewCamera(self.api.open_camera(), self.pictures, self.events)
            exposure, recent = -3, []
            cap.progress('Preparing preview; waiting for exposure to settle...')
            self.api.settle(cap, exposure)
            self.events.put(('ready', None))
            while not self.closing.is_set():
                try:
                    command, value = self.commands.get_nowait()
                except queue.Empty:
                    im = self.api.frame(cap)
                    recent = (recent + [im])[-5:]
                    cap.publish(im, self.api.preview_metrics(recent, exposure))
                    continue
                try:
                    if command == 'exposure':
                        exposure = value
                    elif command == 'scan':
                        exposure, _ = self.api.probe_e0(cap)
                        message = 'E0 found; it will be rechecked before capture.' if exposure is not None else 'No valid E0. Adjust the view or indirect lighting outside capture.'
                        self.events.put(('notice', message))
                        exposure = exposure if exposure is not None else -3
                    elif command == 'capture':
                        room, direction = value
                        path, meta = self.acquire(cap, self.args.dry_run, self.args.scene_id, room, direction)
                        self.events.put(('captured', (path, meta)))
                        exposure = -3
                        if self.args.dry_run:
                            self.closing.set()
                    else:
                        raise ValueError('Unknown interface command')
                except Exception as exc:
                    self.events.put(('notice', 'Could not complete operation: ' + str(exc)))
                    exposure = -3
                if self.closing.is_set():
                    break
                cap.progress('Returning to preview; waiting for exposure to settle...')
                self.api.settle(cap, exposure)
                recent = []
                self.events.put(('ready', None))
        except Exception as exc:
            self.events.put(('fatal', str(exc)))
        finally:
            if cap is not None:
                cap.release()
            self.events.put(('closed', None))


class CaptureWindow:
    def __init__(self, root, worker, args):
        self.root, self.worker, self.args = root, worker, args
        self.busy, self.closing, self.closed = True, False, False
        self.photo = None
        root.title('Exploratory capture · YUY2')
        root.resizable(False, False)
        panel = ttk.Frame(root, padding=10)
        panel.pack(fill='both', expand=True)
        self.room = tk.StringVar(value=args.room or ('dry_run_room' if args.dry_run else ''))
        self.direction = tk.StringVar(value=args.direction or ('dry_run_direction' if args.dry_run else ''))
        labels = ttk.Frame(panel)
        labels.pack(fill='x')
        ttk.Label(labels, text='Room:').pack(side='left')
        self.room_entry = ttk.Entry(labels, textvariable=self.room, width=20)
        self.room_entry.pack(side='left', padx=(4, 12))
        ttk.Label(labels, text='Direction:').pack(side='left')
        self.direction_entry = ttk.Entry(labels, textvariable=self.direction, width=36)
        self.direction_entry.pack(side='left', padx=4)
        self.canvas = tk.Canvas(panel, width=640, height=480, bg='#1c1c1c', highlightthickness=0, takefocus=True)
        self.canvas.pack(pady=8)
        self.image_item = self.canvas.create_image(0, 0, anchor='nw')
        self.canvas.bind('<Button-1>', lambda _: self.canvas.focus_set())
        self.metrics = tk.StringVar(value='Waiting for camera. Preview is not saved.')
        ttk.Label(panel, textvariable=self.metrics, wraplength=640).pack(anchor='w')
        buttons = ttk.Frame(panel)
        buttons.pack(fill='x', pady=7)
        self.actions = []
        for exposure in [-3, -2, -1]:
            button = ttk.Button(buttons, text=f'Preview {exposure}', command=lambda e=exposure: self.submit('exposure', e))
            button.pack(side='left', padx=(0, 4)); self.actions.append(button)
        self.scan_button = ttk.Button(buttons, text='Scan E0', command=lambda: self.submit('scan'))
        self.scan_button.pack(side='left', padx=4); self.actions.append(self.scan_button)
        self.capture_button = ttk.Button(buttons, text='CAPTURE SCENE', command=self.capture)
        self.capture_button.pack(side='left', padx=4); self.actions.append(self.capture_button)
        self.quit_button = ttk.Button(buttons, text='Exit', command=self.close)
        self.quit_button.pack(side='right')
        self.status = tk.StringVar(value='Starting...')
        ttk.Label(panel, textvariable=self.status, wraplength=640).pack(anchor='w')
        self.result = tk.StringVar(value='Enter labels here. Keep the camera, scene and lighting fixed during capture.')
        ttk.Label(panel, textvariable=self.result, wraplength=640, foreground='#174977').pack(anchor='w', pady=4)
        ttk.Label(panel, text='Click image for shortcuts: Space capture · E scan · 1/2/3 exposure · Esc exit').pack(anchor='w')
        root.bind('<KeyPress>', self.key)
        root.protocol('WM_DELETE_WINDOW', self.close)
        self.set_busy(True)
        self.room_entry.configure(state='normal'); self.direction_entry.configure(state='normal')
        self.room_entry.focus_set()
        root.after(30, self.poll)

    def set_busy(self, busy):
        self.busy = busy
        state = 'disabled' if busy or self.closed or self.closing else 'normal'
        for widget in [*self.actions, self.room_entry, self.direction_entry]:
            widget.configure(state=state)

    def submit(self, command, value=None):
        if self.busy or self.closed or self.closing:
            return
        self.set_busy(True)
        self.metrics.set('Operation in progress; displayed frames are not preview QC measurements.')
        self.status.set('CAPTURING: keep the camera, scene and lighting fixed.' if command == 'capture' else 'Preparing exposure...')
        self.worker.commands.put_nowait((command, value))

    def capture(self):
        if self.busy or self.closed or self.closing:
            return
        try:
            room, direction = policy.validate_labels(self.room.get(), self.direction.get())
            if not self.args.dry_run and policy.duplicate_direction(room, direction):
                raise ValueError('This room/view was already accepted. Enter a new label for a new view.')
        except ValueError as exc:
            self.result.set(str(exc)); self.direction_entry.focus_set(); return
        self.submit('capture', (room, direction))

    def key(self, event):
        if event.keysym == 'Escape':
            self.close(); return 'break'
        if isinstance(event.widget, (ttk.Entry, tk.Entry)):
            return  # Spaces and shortcut letters are ordinary text inside labels.
        if event.keysym == 'space':
            self.capture(); return 'break'
        if event.char.lower() == 'e':
            self.submit('scan'); return 'break'
        if event.char in ('1', '2', '3'):
            self.submit('exposure', {'1': -3, '2': -2, '3': -1}[event.char]); return 'break'
        if event.char.lower() == 'q':
            self.close(); return 'break'

    def close(self):
        if self.closed:
            self.root.destroy(); return
        self.closing = True
        self.worker.closing.set()
        self.set_busy(True)
        self.status.set('Closing requested. The camera will close after capture finishes and is saved; do not force the window closed.')

    def poll(self):
        try:
            while True:
                kind, value = self.worker.events.get_nowait()
                if kind == 'progress' and not self.closing:
                    self.status.set(value)
                elif kind == 'notice':
                    self.result.set(value)
                elif kind == 'ready' and not self.closing:
                    self.set_busy(False)
                    self.status.set('Ready. Adjust the view, enter labels and press CAPTURE SCENE.')
                elif kind == 'captured':
                    path, meta = value
                    outcome = 'QC PASS · Scene accepted.' if meta['qc']['passed'] else 'QC FAIL · Scene rejected: ' + '; '.join(meta['qc']['reasons'])
                    if meta['dry_run']:
                        outcome = 'DRY RUN · Excluded from analysis. ' + outcome
                    self.result.set(outcome + '\n' + str(path))
                    if meta['qc']['passed']:
                        self.direction.set('')
                elif kind == 'fatal':
                    self.result.set('Camera error: ' + value)
                    self.metrics.set('Camera stopped. Error recorded; no new capture started.')
                elif kind == 'closed':
                    self.closed = True
                    self.set_busy(True)
                    self.status.set('Camera closed. Press Exit to close the window.')
                    if self.closing:
                        self.root.destroy(); return
        except queue.Empty:
            pass
        try:
            im, metrics = self.worker.pictures.get_nowait()
            self.photo = ImageTk.PhotoImage(Image.fromarray(policy.cv2.cvtColor(im, policy.cv2.COLOR_BGR2RGB)), master=self.root)
            self.canvas.itemconfigure(self.image_item, image=self.photo)
            if metrics is not None and not self.busy:
                e0 = str(metrics['candidate_E0']) if metrics['ready'] else 'not yet valid'
                self.metrics.set(f"YUY2 · Exposure {metrics['exposure']} · E0: {e0}\nLuminance (5 frames): {metrics['luminance']:.3f} [0.35–0.55] · Clipping (%): {100*metrics['clipping']:.2f} (<1%)")
        except queue.Empty:
            pass
        self.root.after(30, self.poll)


def run_gui(args, acquire, probe_e0):
    # No other thread calls camera APIs or mutates the source image buffers.
    from types import SimpleNamespace
    api = SimpleNamespace(open_camera=policy.open_camera, settle=policy.settle,
                          frame=policy.frame, preview_metrics=policy.preview_metrics,
                          probe_e0=probe_e0)
    root = tk.Tk()
    worker = CameraWorker(acquire, args, api)
    CaptureWindow(root, worker, args)
    worker.start()
    root.mainloop()
