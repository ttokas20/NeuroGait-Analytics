import math
import tkinter as tk
from tkinter import filedialog, ttk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates
import cv2
from PIL import Image, ImageTk
import numpy as np
import threading
import time
from queue import Queue
import logging
from scipy.signal import find_peaks

def setup_logger():
    logger = logging.getLogger("ExerciseAnalysisApp")
    if not logger.handlers:  # Only add if not already present
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler("session_log.txt")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)
    return logger


class PersonalizedStepModel:
    @staticmethod
    def calculate_params(height_cm, age, gender, has_parkinsons):
        leg_length_cm = height_cm * 0.54
        base_ratio = 0.76
        
        # Combined factor calculations
        gender_factor = 1.0 if gender.lower() == 'male' else 1.0
        age_factor = {range(0,30): 1.0, range(30,50): 0.98, range(50,65): 0.95, 
                     range(65,75): 0.90}.get(next((r for r in [range(0,30), range(30,50), range(50,65), range(65,75)] if age in r), None), 0.85)
        parkinsons_factor = 0.75 if has_parkinsons else 1.0
        
        base_step_length = max(0, min((leg_length_cm / 100) * base_ratio * gender_factor * age_factor * parkinsons_factor, 1.2))
        variation_factor = 0.3 if has_parkinsons else 0.60
        
        return {
            'leg_length': leg_length_cm,
            'base_step': base_step_length,
            'min_step': max(0, base_step_length * (1 - variation_factor)),
            'max_step': min(2.0, base_step_length * (1 + variation_factor))
        }

class SpeedCalculator:
    def __init__(self, user_config):
        self.logger = logging.getLogger("ExerciseAnalysisApp")
        has_parkinsons = user_config['parkinson_diagnosis'].lower() == 'yes'
        params = PersonalizedStepModel.calculate_params(
            user_config['height_cm'], user_config['age_years'], 
            user_config['gender'], has_parkinsons)
        
        for key, value in params.items():
            setattr(self, key, value)
        
        self.logger.info(f"Speed calculator initialized - Base: {self.base_step:.3f}m, Range: {self.min_step:.3f}-{self.max_step:.3f}m")

    def detect_steps(self, gyro_z_data, timestamps, threshold_factor=0.65, min_distance=10):
        try:
            gyro_z = np.array(gyro_z_data)
            valid_mask = ~np.isnan(gyro_z)
            gyro_z, valid_timestamps = gyro_z[valid_mask], timestamps[valid_mask] if hasattr(timestamps, '__getitem__') else timestamps
            
            if len(gyro_z) == 0:
                return [], [], []
            
            threshold = max(np.mean(np.abs(gyro_z)) * threshold_factor, np.std(gyro_z) * 0.65)
            peaks, _ = find_peaks(np.abs(gyro_z), height=threshold, distance=min_distance, prominence=threshold*0.65)
            
            step_times = [valid_timestamps.iloc[i] if hasattr(valid_timestamps, 'iloc') else valid_timestamps[i] 
                         for i in peaks if i < len(valid_timestamps)]
            peak_magnitudes = [abs(gyro_z[i]) for i in peaks]
            
            self.logger.info(f"Detected {len(peaks)} steps")
            return peaks, step_times, peak_magnitudes
        except Exception as e:
            self.logger.error(f"Step detection error: {e}")
            return [], [], []

    def calculate_dynamic_step_length(self, peak_magnitudes):
        if not peak_magnitudes:
            return []
        magnitudes = np.array(peak_magnitudes)
        mag_mean = np.mean(magnitudes)
        mag_std = np.std(magnitudes)
        # Avoid division by zero (if all peaks are similar)
        if mag_std < 1e-6:
            return [self.base_step] * len(magnitudes)
        # Z-score normalization
        normalized_mags = (magnitudes - mag_mean) / mag_std
        # LINEAR mapping with a calibrated coefficient (adjust 0.15 as you see fit)
        step_lengths = self.base_step + 0.15 * normalized_mags
        # Restrict to personalized min/max
        step_lengths = np.clip(step_lengths, self.min_step, self.max_step)
        return step_lengths.tolist()


    def calculate_speed(self, df, gyro_z_columns, timestamp_col='TimeStamp'):
        try:
            all_step_data = []
            for col in gyro_z_columns:
                if col in df.columns:
                    _, step_times, peak_magnitudes = self.detect_steps(df[col].fillna(0).values, df[timestamp_col])
                    step_lengths = self.calculate_dynamic_step_length(peak_magnitudes)
                    for step_time, step_length in zip(step_times, step_lengths):
                        frame_idx = np.abs(df[timestamp_col] - step_time).idxmin()
                        all_step_data.append((step_time, step_length, frame_idx))
            
            all_step_data.sort()
            if len(all_step_data) < 2:
                return pd.DataFrame({timestamp_col: df[timestamp_col], 'Speed_cms': np.zeros(len(df)), 'StepLength_cm': np.zeros(len(df))})
            
            speeds_cms = np.zeros(len(df))
            step_lengths_cm = np.zeros(len(df))
            
            for i, (step_time, step_length_m, frame_idx) in enumerate(all_step_data):
                current_step_length_cm = step_length_m * 100
                if i > 0:
                    prev_step_time, prev_frame_idx = all_step_data[i-1][0], all_step_data[i-1][2]
                    time_diff = (step_time - prev_step_time).total_seconds()
                    if time_diff > 0:
                        speed_cms = current_step_length_cm / time_diff
                        for frame in range(prev_frame_idx, min(frame_idx + 1, len(df))):
                            speeds_cms[frame] = speed_cms
                            step_lengths_cm[frame] = current_step_length_cm
                
                if frame_idx < len(df):
                    step_lengths_cm[frame_idx] = current_step_length_cm
            
            # Forward fill and smooth
            last_step_length = 0.0
            for i in range(len(step_lengths_cm)):
                if step_lengths_cm[i] > 0:
                    last_step_length = step_lengths_cm[i]
                elif last_step_length > 0:
                    step_lengths_cm[i] = last_step_length
            
            speeds_cms = self.smooth_data(speeds_cms.tolist())
            speeds_cms = [min(max(s, 0), 300) for s in speeds_cms]
            step_lengths_cm = [min(max(s, 0), 200) for s in step_lengths_cm]
            
            return pd.DataFrame({timestamp_col: df[timestamp_col], 'Speed_cms': speeds_cms, 'StepLength_cm': step_lengths_cm})
        except Exception as e:
            self.logger.error(f"Speed calculation error: {e}")
            return pd.DataFrame({timestamp_col: df[timestamp_col], 'Speed_cms': np.zeros(len(df)), 'StepLength_cm': np.zeros(len(df))})

    def smooth_data(self, data, window=5):
        if len(data) < window:
            return data
        return [np.mean(data[max(0, i - window//2):min(len(data), i + window//2 + 1)]) for i in range(len(data))]

class UserConfigWindow:
    def __init__(self, parent, callback):
        self.parent, self.callback, self.window = parent, callback, None
        self.vars = {
            'parkinson_diagnosis': tk.StringVar(value="No"),
            'height_cm': tk.StringVar(value="190.5"),
            'gender': tk.StringVar(value="Male"),
            'age_years': tk.StringVar(value="23")
        }

    def show(self):
        if self.window and self.window.winfo_exists():
            self.window.lift()
            return
        
        self.window = tk.Toplevel(self.parent)
        self.window.title("User Configuration")
        self.window.geometry("400x450")
        self.window.transient(self.parent)
        self.window.grab_set()
        
        main_frame = tk.Frame(self.window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Create form sections
        sections = [
            ("Parkinson's Diagnosis", [
                (tk.Radiobutton, {"text": "Yes", "variable": self.vars['parkinson_diagnosis'], "value": "Yes"}),
                (tk.Radiobutton, {"text": "No", "variable": self.vars['parkinson_diagnosis'], "value": "No"})
            ]),
            ("Height", [(self.create_input_field, {"var": self.vars['height_cm'], "unit": "cm", "validator": lambda e: self.validate_field(self.vars['height_cm'], "170.0", True, 50.0, 250.0)})]),
            ("Age", [(self.create_input_field, {"var": self.vars['age_years'], "unit": "years", "validator": lambda e: self.validate_field(self.vars['age_years'], "30", False, 1, 120)})]),
            ("Gender", [
                (tk.Radiobutton, {"text": "Male", "variable": self.vars['gender'], "value": "Male"}),
                (tk.Radiobutton, {"text": "Female", "variable": self.vars['gender'], "value": "Female"})
            ])
        ]
        
        for title, widgets in sections:
            frame = tk.LabelFrame(main_frame, text=title, font=('TkDefaultFont', 10, 'bold'))
            frame.pack(fill=tk.X, pady=(0, 15))
            for widget_class, kwargs in widgets:
                if widget_class == self.create_input_field:
                    self.create_input_field(frame, **kwargs)
                else:
                    widget_class(frame, **kwargs).pack(anchor='w', padx=10, pady=5)
        
        # Buttons
        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(15, 0))
        tk.Button(button_frame, text="Cancel", command=self.cancel, bg='lightgreen', width=12, font=('TkDefaultFont', 10, 'bold')).pack(side=tk.LEFT, padx=(0, 10))
        tk.Button(button_frame, text="Save", command=self.save_config, width=12).pack(side=tk.LEFT)

    def create_input_field(self, parent, var, unit, validator):
        input_frame = tk.Frame(parent)
        input_frame.pack(pady=10)
        entry = tk.Entry(input_frame, textvariable=var, width=10, justify='center')
        entry.pack(side=tk.LEFT, padx=(10, 5))
        tk.Label(input_frame, text=unit).pack(side=tk.LEFT)
        entry.bind('<FocusOut>', validator)

    def validate_field(self, var, default, is_float=True, min_val=1, max_val=250):
        value = var.get().strip()
        if not value:
            var.set(default)
            return
        
        try:
            if is_float:
                cleaned = ''.join(c for c in value if c.isdigit() or c == '.')
                if cleaned.count('.') > 1:
                    parts = cleaned.split('.')
                    cleaned = parts[0] + '.' + ''.join(parts[1:])
                if '.' in cleaned and len(cleaned.split('.')[1]) > 1:
                    integer_part, decimal_part = cleaned.split('.')
                    cleaned = integer_part + '.' + decimal_part[:1]
                val = float(cleaned)
                var.set(f"{max(min_val, min(val, max_val)):.1f}")
            else:
                cleaned = ''.join(c for c in value if c.isdigit())
                if cleaned:
                    val = int(cleaned)
                    var.set(str(max(min_val, min(val, max_val))))
                else:
                    var.set(default)
        except (ValueError, IndexError):
            var.set(default)

    def save_config(self):
        try:
            config = {key: float(var.get()) if key in ['height_cm'] else int(var.get()) if key == 'age_years' else var.get() 
                     for key, var in self.vars.items()}
        except ValueError:
            config = {'parkinson_diagnosis': 'No', 'height_cm': 182.8, 'age_years': 20, 'gender': 'Male'}
        
        self.callback(config)
        self.window.destroy()

    def cancel(self):
        self.window.destroy()

class ExerciseAnalysisApp:
    def __init__(self, root):
        self.logger = setup_logger()
        self.logger.info("Application started")
        self.root = root
        root.title("Exercise Analysis Dashboard")
        root.geometry("1200x700")
        root.minsize(1000, 600)
        
        # Initialize all attributes
        self.video_path = ""
        self.vid_cap = None
        self.df1 = self.df2 = self.df = None
        self.playing = False
        self.current_frame = 0
        self.video_fps = 60
        self.video_duration = 0
        self.frame_queue = Queue(maxsize=5)
        self.stop_threads = False
        self.last_frame_time = self.frame_delay = 0
        self.available_columns = []
        self.column_groups = []
        self.axes = []
        self.lines = []
        self.step_length_text = None
        self.speed_calculator = None
        self.speed_data = None
        self.user_config = {'parkinson_diagnosis': 'No', 'height_cm': 190.5, 'age_years': 23, 'gender': 'Male'}
        
        self.config_window = UserConfigWindow(self.root, self.update_user_config)
        self.create_widgets()
        self.update_speed_calculator()

    def update_speed_calculator(self):
        self.speed_calculator = SpeedCalculator(self.user_config)
        self.speed_data = None

    def update_user_config(self, config):
        self.user_config.update(config)
        params = PersonalizedStepModel.calculate_params(
            config['height_cm'], config['age_years'], 
            config['gender'], config['parkinson_diagnosis'].lower() == 'yes')
        
        self.status_var.set(f"User: {config['gender']}, {config['age_years']}yrs, {config['height_cm']}cm, "
                           f"Leg: {params['leg_length']:.1f}cm, Base step: {params['base_step']:.2f}m, "
                           f"Parkinson's: {config['parkinson_diagnosis']}")
        self.logger.info(f"User configuration updated: {self.user_config}")
        self.update_speed_calculator()

    def create_widgets(self):
        # Control frame
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        buttons = [
            ("User Data", self.config_window.show),
            ("Upload Video", self.load_video),
            ("Upload Left Leg Data", lambda: self.load_csv(1)),
            ("Upload Right Leg Data", lambda: self.load_csv(2)),
            ("Select Column Groups", self.show_dropdown),
            ("Save Graph", self.save_complete_graph)
        ]
        
        for text, command in buttons:
            tk.Button(control_frame, text=text, command=command).pack(side=tk.LEFT, padx=5)
        
        self.play_btn = tk.Button(control_frame, text="Play", command=self.toggle_play)
        self.play_btn.pack(side=tk.LEFT, padx=5)
        
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(control_frame, variable=self.progress_var, maximum=100)
        self.progress.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        
        # Content frame
        content_frame = tk.Frame(self.root)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Video frame
        self.video_frame = tk.Frame(content_frame, width=300, height=600)
        self.video_frame.pack_propagate(False)
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.video_panel = tk.Label(self.video_frame)
        self.video_panel.pack(fill=tk.BOTH, expand=True)
        
        # Graph frame
        graph_frame = tk.Frame(content_frame, width=800, height=600)
        graph_frame.pack_propagate(False)
        graph_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.fig = plt.Figure(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def save_complete_graph(self):
        if not self.column_groups or self.df is None:
            self.status_var.set("No data to save - please load data and select column groups first")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("PDF files", "*.pdf"), ("All files", "*.*")],
            title="Save complete graph as"
        )
        
        if file_path:
            try:
                save_fig = plt.Figure(figsize=(12, 8), dpi=300)
                overall_avg_step_length = None
                
                if self.speed_data is not None and 'StepLength_cm' in self.speed_data.columns:
                    non_zero_steps = self.speed_data['StepLength_cm'][self.speed_data['StepLength_cm'] > 0]
                    if len(non_zero_steps) > 0:
                        overall_avg_step_length = np.mean(non_zero_steps)
                
                for i, group in enumerate(self.column_groups):
                    ax = save_fig.add_subplot(len(self.column_groups), 1, i + 1)
                    for col in group:
                        data = self.get_data_for_column(col)
                        if data is not None:
                            ax.plot(self.df['TimeStamp'], data, label=col, linewidth=1.5)
                    
                    ax.legend(loc='upper left', fontsize=10)
                    ax.set_ylabel(f"{group}", fontsize=12)
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                    ax.tick_params(axis='both', labelsize=10)
                    ax.grid(True, alpha=0.3)
                    
                    if 'StepLength_cm' in group and overall_avg_step_length is not None:
                        ax.text(0.98, 0.95, f'Average Step Length: {overall_avg_step_length:.1f} cm',
                               transform=ax.transAxes, fontsize=14, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8, edgecolor='darkgreen'),
                               verticalalignment='top', horizontalalignment='right')
                    
                    if i == len(self.column_groups) - 1:
                        ax.set_xlabel("Time", fontsize=12)
                
                save_fig.suptitle("Exercise Analysis - Complete Dataset", fontsize=16, fontweight='bold')
                save_fig.tight_layout()
                save_fig.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
                plt.close(save_fig)
                self.status_var.set(f"Complete graph saved to {file_path}")
            except Exception as e:
                self.status_var.set(f"Error saving complete graph: {str(e)}")

    def get_gyro_z_columns(self):
        if self.df is None:
            return []
        return [col for col in self.df.columns 
                if any(pattern in col.lower() for pattern in ['gz', 'gyro']) and 
                ('z' in col.lower() or col.upper().endswith('GZ') or col.endswith(('-L', '-R')))]

    def calculate_speed_data(self):
        if self.df is None or self.speed_calculator is None:
            return
        gyro_z_columns = self.get_gyro_z_columns()
        if gyro_z_columns:
            self.speed_data = self.speed_calculator.calculate_speed(self.df, gyro_z_columns, 'TimeStamp')

    def calculate_angles(self):
        # Process original dataframes separately BEFORE merging
        for df_original, suffix in [(self.df1, '-L'), (self.df2, '-R')]:
            if df_original is None or df_original.empty:
                continue
                
            # Add TimeInSec if not present in original dataframe
            if 'TimeInSec' not in df_original.columns:
                df_original['TimeInSec'] = (df_original['TimeStamp'] - df_original['TimeStamp'].iloc[0]).dt.total_seconds()

            # Calculate time difference for original dataframe
            df_original['time_diff'] = df_original['TimeInSec'].diff().fillna(0)

            # Calculate integrated gyroscope Y (theta_y) using original data
            gy_col = f'GY{suffix}'
            if gy_col in df_original.columns:
                # Convert to numeric and fill NaN
                df_original[gy_col] = pd.to_numeric(df_original[gy_col], errors='coerce').fillna(0)
                
                # Calculate integrated angle using original data
                df_original[f'theta_y{suffix}'] = (df_original[gy_col] * df_original['time_diff']).cumsum()

            # Calculate roll angle from accelerometer using original data
            ax_col = f'AX{suffix}'
            ay_col = f'AY{suffix}'
            az_col = f'AZ{suffix}'
            
            if all(col in df_original.columns for col in [ax_col, ay_col, az_col]):
                # Convert to numeric and fill NaN
                for col in [ax_col, ay_col, az_col]:
                    df_original[col] = pd.to_numeric(df_original[col], errors='coerce').fillna(0)

                # Calculate roll using vectorized operations on original data
                ax_vals = df_original[ax_col].values
                ay_vals = df_original[ay_col].values
                az_vals = df_original[az_col].values

                # Roll calculation formula
                roll_radians = np.arctan2(ay_vals, np.sqrt(ax_vals**2 + az_vals**2))
                roll_degrees = np.degrees(roll_radians)
                roll_degrees = np.where(az_vals > 0, np.abs(roll_degrees), 180 - np.abs(roll_degrees))

                df_original[f'roll{suffix}'] = roll_degrees

        # Now re-merge the dataframes with the calculated angles
        if self.df1 is not None and self.df2 is not None:
            try:
                self.df = pd.merge_asof(self.df1.sort_values("TimeStamp"), self.df2.sort_values("TimeStamp"),
                                    on="TimeStamp", direction="nearest")
            except Exception as e:
                self.status_var.set(f"Merge error: {e}")
                return
        elif self.df1 is not None:
            self.df = self.df1.copy()
        else:
            return

        # Force numeric conversion for all non-Timestamp columns
        for col in self.df.columns:
            if col != 'TimeStamp':
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)

    def show_dropdown(self):
        if not self.available_columns:
            return
        
        if hasattr(self, 'dropdown_win') and self.dropdown_win.winfo_exists():
            self.dropdown_win.lift()
            return
        
        self.dropdown_win = tk.Toplevel(self.root)
        self.dropdown_win.title("Select Column Groups")
        self.dropdown_win.geometry("320x600")
        self.dropdown_win.transient(self.root)
        self.dropdown_win.grab_set()
        
        container = tk.Frame(self.dropdown_win)
        container.pack(fill=tk.BOTH, expand=True)
        
        canvas = tk.Canvas(container)
        scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        group_vars = []
        
        def add_group():
            var_dict = {}
            frame = tk.LabelFrame(scrollable_frame, text=f"Group {len(group_vars) + 1}")
            frame.pack(fill=tk.X, padx=5, pady=5)
            
            for col in self.available_columns:
                var = tk.BooleanVar()
                var_dict[col] = var
                tk.Checkbutton(frame, text=col, variable=var).pack(anchor='w')
            
            gyro_z_cols = self.get_gyro_z_columns()
            if gyro_z_cols:
                tk.Label(frame, text="-- Speed Values ---", font=('TkDefaultFont', 9, 'bold')).pack(anchor='w', pady=(5,0))
                for col, text, color in [('Speed_cms', 'Speed (cm/s) - Personalized', 'blue'),
                                       ('StepLength_cm', 'Step Length (cm) - Personalized', 'green')]:
                    var = tk.BooleanVar()
                    var_dict[col] = var
                    tk.Checkbutton(frame, text=text, variable=var, fg=color).pack(anchor='w')
            
            # Add angle calculation options
            tk.Label(frame, text="--- Angle Calculations ---", font=('TkDefaultFont', 9, 'bold')).pack(anchor='w', pady=(5,0))
            for col, text, color in [('theta_y-L', 'Integrated Gyro Y (Left)', 'red'),
                                   ('theta_y-R', 'Integrated Gyro Y (Right)', 'red'),
                                   ('roll-L', 'Roll Angle (Left)', 'purple'),
                                   ('roll-R', 'Roll Angle (Right)', 'purple')]:
                var = tk.BooleanVar()
                var_dict[col] = var
                tk.Checkbutton(frame, text=text, variable=var, fg=color).pack(anchor='w')
            
            group_vars.append(var_dict)
        
        def apply_groups():
            if any(any(col in ['Speed_cms', 'StepLength_cm'] and var.get() for col, var in group.items()) for group in group_vars) and self.speed_data is None:
                self.calculate_speed_data()
            
            self.column_groups = [[col for col, var in group.items() if var.get()] 
                                for group in group_vars if any(var.get() for var in group.values())]
            self.column_changed()
            self.dropdown_win.destroy()
        
        tk.Button(self.dropdown_win, text="Apply", command=apply_groups).pack(pady=5)
        tk.Button(self.dropdown_win, text="New Plot", command=add_group).pack(pady=5)
        add_group()

    def load_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
        if self.video_path:
            if self.vid_cap:
                self.vid_cap.release()
                self.frame_queue.queue.clear()
            
            self.vid_cap = cv2.VideoCapture(self.video_path)
            if not self.vid_cap.isOpened():
                self.status_var.set("Error: Could not open video file")
                return
            
            self.video_fps = self.vid_cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_duration = frame_count / self.video_fps
            self.frame_delay = 1 / (self.video_fps * 2.0)
            
            self.status_var.set(f"Loaded video: {self.video_path} (Duration: {self.video_duration:.2f}s)")
            self.current_frame = 0
            self.progress_var.set(0)
            self.stop_threads = False
            threading.Thread(target=self.video_reader_thread, daemon=True).start()

    def video_reader_thread(self):
        while not self.stop_threads and self.vid_cap.isOpened():
            if self.frame_queue.qsize() < 20:
                ret, frame = self.vid_cap.read()
                if ret:
                    self.frame_queue.put(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                else:
                    self.vid_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.current_frame = 0
            else:
                time.sleep(0.001)

    def load_csv(self, csv_num):
        path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if path:
            try:
                df = pd.read_csv(path)
                df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
                suffix = '-L' if csv_num == 1 else '-R'
                df.rename(columns={col: f"{col}{suffix}" for col in df.columns if col != 'TimeStamp'}, inplace=True)
                
                if csv_num == 1:
                    self.df1 = df
                else:
                    self.df2 = df
                
                self.status_var.set(f"Loaded CSV {csv_num}: {path}")
                self.merge_dataframes()
            except Exception as e:
                self.status_var.set(f"Error loading CSV {csv_num}: {str(e)}")

    def merge_dataframes(self):
        if self.df1 is not None and self.df2 is not None:
            try:
                self.df = pd.merge_asof(self.df1.sort_values("TimeStamp"), self.df2.sort_values("TimeStamp"),
                                      on="TimeStamp", direction="nearest", tolerance=pd.Timedelta("50ms"))
            except Exception as e:
                self.status_var.set(f"Merge error: {e}")
                return
        elif self.df1 is not None:
            self.df = self.df1.copy()
        else:
            return
        
        # Force numeric conversion for all non-Timestamp columns
        for col in self.df.columns:
            if col != 'TimeStamp':
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)
        
        # Calculate angles
        self.calculate_angles()
        
        # Update available columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        # Define columns to exclude from dropdown
        excluded_columns = {
            "TimeStamp", "Leg-R", "TimeInSec", "time_diff", 
            "theta_y-L", "roll-L", "theta_y-R", "roll-R", "Unnamed: 7-L", "Unnamed: 7-R"
        }
        self.available_columns = [col for col in numeric_cols if col not in excluded_columns]
        self.speed_data = None

    def get_data_for_column(self, col):
        if col in ['Speed_cms', 'StepLength_cm'] and self.speed_data is not None:
            return self.speed_data[col]
        return self.df.get(col)

    def column_changed(self):
        if not self.column_groups:
            self.status_var.set("Please define at least one group of columns")
            return
        
        self.fig.clf()
        self.axes = []
        self.lines = []
        self.step_length_text = None
        
        for i, group in enumerate(self.column_groups):
            ax = self.fig.add_subplot(len(self.column_groups), 1, i + 1)
            self.axes.append(ax)
            group_lines = []
            
            for col in group:
                data = self.get_data_for_column(col)
                if data is not None:
                    line, = ax.plot([], [], label=col)
                    group_lines.append(line)
            
            ax.legend(loc='upper left')
            ax.set_ylabel(f"{group}")
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            
            if 'StepLength_cm' in group:
                self.step_length_text = ax.text(0.98, 0.95, '', transform=ax.transAxes,
                                              fontsize=12, fontweight='bold',
                                              bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8),
                                              verticalalignment='top', horizontalalignment='right')
            
            if i == len(self.column_groups) - 1:
                ax.set_xlabel("Time")
            
            self.lines.append(group_lines)
        
        self.fig.tight_layout()
        self.canvas.draw()

    def update_graph_frame(self, frame_idx):
        if self.df is None or frame_idx >= len(self.df):
            return
        
        start = max(0, frame_idx - 300)
        end = frame_idx + 1
        time_slice = self.df['TimeStamp'][start:end]
        
        avg_step_length = None
        if self.speed_data is not None and 'StepLength_cm' in self.speed_data.columns:
            step_lengths_so_far = self.speed_data['StepLength_cm'][:frame_idx + 1]
            non_zero_steps = step_lengths_so_far[step_lengths_so_far > 0]
            if len(non_zero_steps) > 0:
                avg_step_length = np.mean(non_zero_steps)
        
        for ax, group_lines, group in zip(self.axes, self.lines, self.column_groups):
            y_values = []
            for line, col in zip(group_lines, group):
                data = self.get_data_for_column(col)
                if data is not None:
                    y = data[start:end]
                    line.set_data(time_slice, y)
                    y_values.extend(y)
            
            if y_values:
                ax.set_xlim(time_slice.iloc[0], time_slice.iloc[-1])
                ymin, ymax = min(y_values), max(y_values)
                y_range = ymax - ymin
                padding = y_range * 0.1 if y_range > 0 else 1
                ax.set_ylim(ymin - padding, ymax + padding)
            
            if hasattr(self, 'step_length_text') and self.step_length_text is not None:
                text = f'Avg Step Length: {avg_step_length:.1f} cm' if avg_step_length and avg_step_length > 0 else 'Avg Step Length: -- cm'
                self.step_length_text.set_text(text)
        
        self.canvas.draw()

    def toggle_play(self):
        if not self.vid_cap:
            self.status_var.set("Please load a video first")
            return
        
        self.playing = not self.playing
        self.play_btn.config(text="Pause" if self.playing else "Play")
        
        if self.playing:
            self.last_frame_time = time.time()
            self.update_video_frame_loop()

    def update_video_frame_loop(self):
        if not self.playing or self.stop_threads:
            return
        
        current_time = time.time()
        elapsed = current_time - self.last_frame_time
        
        if elapsed >= self.frame_delay and not self.frame_queue.empty():
            frame = self.frame_queue.get()
            self.current_frame += 1
            self.last_frame_time = current_time
            
            progress = (self.current_frame / int(self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))) * 100
            self.progress_var.set(progress)
            
            self.update_video_frame(frame)
            self.update_graph_frame(self.current_frame)
        
        self.root.after(int(self.frame_delay * 1000), self.update_video_frame_loop)

    def update_video_frame(self, frame):
        if frame is not None:
            img = Image.fromarray(frame)
            panel_width = self.video_frame.winfo_width()
            panel_height = self.video_frame.winfo_height()
            
            img_ratio = img.width / img.height
            panel_ratio = panel_width / panel_height
            
            if panel_ratio > img_ratio:
                new_height = panel_height
                new_width = int(img_ratio * new_height)
            else:
                new_width = panel_width
                new_height = int(new_width / img_ratio)
            
            img = img.resize((new_width, new_height), Image.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_panel.imgtk = imgtk
            self.video_panel.config(image=imgtk)

    def on_closing(self):
        self.stop_threads = True
        if self.vid_cap:
            self.vid_cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ExerciseAnalysisApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()