"""
C-Scan Composite Analysis - Modern NiceGUI Version
Advanced visualization and analysis of composite C-scan data with a beautiful web interface.
FINAL CORRECTED VERSION - CLOUD DEPLOYMENT READY
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import re
from scipy.ndimage import zoom
import json
import difflib
import asyncio
from pathlib import Path
import base64
from io import BytesIO, StringIO
import sys
import tempfile
import shutil
# from tkinter import filedialog, Tk # No longer needed for web deployment

from nicegui import ui, app
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for web

# Import all the original analysis functions
from typing import Optional, Tuple, List, Dict, Any

# --- Application State ---
app_state = {
    'composite': None,
    'vmin': 0,
    'vmax': 0,
    'colourised': True,
    'min_thickness': 0.0,
    'max_thickness': 0.0,
    'scan_regions': None,
    'mapped_files': [],
    'x_offset': 0,
    'y_offset': 0,
    'csv_file': None, # To store uploaded CSV
    'cscan_files': [] # To store uploaded C-Scan files
}

# --- Settings and Data Paths ---
SETTINGS_PATH = os.path.join(os.path.expanduser('~'), '.cscan_composite_settings.json')
ADJUSTMENTS_PATH = os.path.join(os.path.expanduser('~'), '.cscan_composite_adjustments.json')

# Global variables for scan adjustments and folder memory
scan_adjustments = {}
last_single_scan_folder = ''


# --- Core Data Processing Functions ---

def parse_matrix_from_file(file_source: Any) -> np.ndarray:
    """Parse matrix data from C-scan file path or file-like object."""
    lines = []
    if isinstance(file_source, str):
        with open(file_source, "r") as f:
            lines = f.readlines()
    else:  # Assumes a file-like object with bytes content
        try:
            content_str = file_source.read().decode('utf-8')
            lines = StringIO(content_str).readlines()
        except Exception:
            # Try another common encoding if utf-8 fails
            file_source.seek(0)
            content_str = file_source.read().decode('latin1')
            lines = StringIO(content_str).readlines()

    matrix_data = []
    reading_matrix = False
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        try:
            float(parts[0])
            reading_matrix = True
        except ValueError:
            if reading_matrix:
                break
            continue
        if reading_matrix:
            row = [float(val) if val != "---" else np.nan for val in parts[1:]]
            matrix_data.append(row)

    if not matrix_data:
        return np.array([[]])
        
    max_cols = max(len(r) for r in matrix_data) if matrix_data else 0
    padded_matrix = [r + [np.nan] * (max_cols - len(r)) for r in matrix_data]
    return np.array(padded_matrix)

def extract_strake_shell_scan(name: str) -> Tuple[Optional[int], Optional[int]]:
    """Extract strake and shell/scan numbers"""
    strake, shell_or_scan = None, None
    m_strake = re.search(r"strake\s*(\d+)", name, re.IGNORECASE)
    if m_strake:
        strake = int(m_strake.group(1))
    m_shell = re.search(r"(?:shell|scan)\s*(\d+)", name, re.IGNORECASE)
    if m_shell:
        shell_or_scan = int(m_shell.group(1))
    return strake, shell_or_scan

def clean_scan_name(scan_name: str) -> str:
    """Clean and normalize the scan name for consistent matching"""
    return re.sub(r'\s+', ' ', scan_name.lower().strip())

def fuzzy_match_best_mapping(scan_names: List[str], files_in_folder: List[str]) -> Dict[str, str]:
    """Find best mapping between scan names and files"""
    candidates = {}
    cleaned_files = {fname: clean_scan_name(fname) for fname in files_in_folder}

    for scan_name in scan_names:
        scan_strake, scan_shellscan = extract_strake_shell_scan(scan_name)
        candidates[scan_name] = []
        cleaned_scan = clean_scan_name(scan_name)

        for fname, cleaned_fname in cleaned_files.items():
            ratio = difflib.SequenceMatcher(None, cleaned_scan, cleaned_fname).ratio()
            file_strake, file_shellscan = extract_strake_shell_scan(fname)
            if scan_strake and file_strake and scan_strake == file_strake:
                ratio += 0.3
            if scan_shellscan and file_shellscan and scan_shellscan == file_shellscan:
                ratio += 0.2
            candidates[scan_name].append((ratio, fname))
        
        candidates[scan_name].sort(reverse=True)

    assigned_files = set()
    mapping = {}
    scan_names_sorted = sorted(scan_names, key=lambda n: -candidates[n][0][0] if candidates.get(n) else 0)

    for scan_name in scan_names_sorted:
        for _, fname in candidates.get(scan_name, []):
            if fname not in assigned_files:
                mapping[scan_name] = fname
                assigned_files.add(fname)
                break
    return mapping

def transform_scan_for_physical_orientation(mat, apply_correction=True):
    """Transform scan data to match physical scanning orientation (bottom-up, right-to-left)"""
    return np.flip(mat, axis=(0, 1)) if apply_correction else mat

def get_corrected_placement_coordinates(row, mat_shape):
    """Calculate placement coordinates from CSV data"""
    height, width = mat_shape
    x_start = int(np.round(row["X Start (mm)"]))
    y_start = int(np.round(row["Y Start (mm)"]))
    x_end = x_start + width - 1
    y_end = y_start + height - 1
    return x_start, y_start, x_end, y_end
    
def build_composite(csv_path, cscan_folder, colourised, strake_filter, flip_scans, scanning_correction):
    """Build composite C-scan from CSV and scan files"""
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns] # Clean column names
    
    required_cols = ["File Name", "X Start (mm)", "Y Start (mm)"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV must contain the following columns: {required_cols}")

    df = df[required_cols].dropna().reset_index(drop=True)

    if strake_filter:
        df = df[df["File Name"].str.contains(fr"strake\s*{strake_filter}", case=False, na=False, regex=True)]

    if df.empty:
        raise RuntimeError("No scans found in CSV after applying filters.")

    files_in_folder = os.listdir(cscan_folder)
    scan_names = [str(row["File Name"]).strip() for _, row in df.iterrows()]
    best_mapping = fuzzy_match_best_mapping(scan_names, files_in_folder)

    all_coords = []
    temp_mats = {}

    # First pass: parse all matrices to determine canvas size
    for idx, row in df.iterrows():
        scan_name = str(row["File Name"]).strip()
        matched_file = best_mapping.get(scan_name)
        if not matched_file:
            continue
        
        mat = parse_matrix_from_file(os.path.join(cscan_folder, matched_file))
        if mat.size == 0 or np.isnan(mat).all():
            continue

        temp_mats[scan_name] = mat
        x_start, y_start, _, _ = get_corrected_placement_coordinates(row, mat.shape)
        all_coords.extend([(x_start, y_start), (x_start + mat.shape[1], y_start + mat.shape[0])])

    if not all_coords:
        raise RuntimeError("No valid scans could be loaded to determine composite dimensions.")

    x_coords, y_coords = zip(*all_coords)
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    canvas_width = int(np.ceil(x_max - x_min))
    canvas_height = int(np.ceil(y_max - y_min))
    x_offset = int(-x_min)
    y_offset = int(-y_min)

    composite = np.full((canvas_height, canvas_width), np.nan)
    overall_min, overall_max = np.inf, -np.inf
    mapped_files, missing_scans = [], []

    # Second pass: place matrices on the canvas
    for idx, row in df.iterrows():
        scan_name = str(row["File Name"]).strip()
        if scan_name not in temp_mats:
            missing_scans.append(scan_name)
            continue
        
        mapped_files.append((scan_name, best_mapping[scan_name]))
        mat = temp_mats[scan_name]

        if scanning_correction:
            mat = transform_scan_for_physical_orientation(mat)
        if flip_scans:
            mat = np.flipud(mat)

        height, width = mat.shape
        x_start, y_start, _, _ = get_corrected_placement_coordinates(row, (height, width))
        
        x0, y0 = x_start + x_offset, y_start + y_offset
        
        # Ensure placement is within canvas bounds
        if y0 + height > composite.shape[0] or x0 + width > composite.shape[1]:
            continue

        region = composite[y0:y0+height, x0:x0+width]
        mat_region = mat
        
        region_valid = ~np.isnan(region)
        mat_valid = ~np.isnan(mat_region)

        copy_mask = ~region_valid & mat_valid
        region[copy_mask] = mat_region[copy_mask]

        avg_mask = region_valid & mat_valid
        region[avg_mask] = (region[avg_mask] + mat_region[avg_mask]) / 2.0

        composite[y0:y0+height, x0:x0+width] = region
        
        scan_min, scan_max = np.nanmin(mat), np.nanmax(mat)
        if not np.isinf(scan_min): overall_min = min(overall_min, scan_min)
        if not np.isinf(scan_max): overall_max = max(overall_max, scan_max)

    if not mapped_files:
        raise RuntimeError("No valid scans were loaded into the composite.")
        
    return composite, overall_min, overall_max, mapped_files, missing_scans, x_offset, y_offset


def plot_to_base64(composite, min_val, max_val, colourised, x_offset, y_offset):
    """Convert matplotlib plot to base64 string for web display"""
    fig, ax = plt.subplots(figsize=(16, 9), facecolor='#00000000')
    ax.set_facecolor('#111111DD')

    if np.isnan(composite).all():
        ax.text(0.5, 0.5, "No Data to Display", color='white', ha='center', va='center', fontsize=20)
        ax.set_axis_off()
    else:
        cmap = "gray"
        if colourised:
            colors = [(1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1)]
            positions = [0.0, 0.33, 0.66, 1.0]
            cmap = LinearSegmentedColormap.from_list("custom_cmap", list(zip(positions, colors)))

        height, width = composite.shape
        extent = [-x_offset, width - x_offset, -y_offset, height - y_offset]
        
        im = ax.imshow(composite, cmap=cmap, origin='lower', vmin=min_val, vmax=max_val, aspect='equal', extent=extent)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Thickness (mm)", color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

        ax.set_title("Composite C-Scan Heatmap", color='white')
        ax.set_xlabel("X (mm)", color='white')
        ax.set_ylabel("Y (mm)", color='white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('white')

    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150, transparent=True)
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


# --- Settings Management ---
def load_settings():
    """Load saved settings"""
    if not os.path.exists(SETTINGS_PATH):
        return '', '', 0.0, 0.0
    try:
        with open(SETTINGS_PATH, 'r') as f:
            data = json.load(f)
            return (data.get('csv_path', ''), data.get('cscan_folder', ''),
                    data.get('min_thickness', 0.0), data.get('max_thickness', 0.0))
    except (json.JSONDecodeError, IOError):
        return '', '', 0.0, 0.0

def save_settings(csv_path, cscan_folder, min_thickness, max_thickness):
    """Save current settings"""
    settings = {
        'csv_path': csv_path, 'cscan_folder': cscan_folder,
        'min_thickness': min_thickness, 'max_thickness': max_thickness
    }
    try:
        with open(SETTINGS_PATH, 'w') as f:
            json.dump(settings, f, indent=4)
    except IOError as e:
        print(f"Error saving settings: {e}")


# --- UI Callbacks ---
def handle_csv_upload(e: ui.events.UploadEventArguments):
    """Handle the upload of the CSV file."""
    app_state['csv_file'] = {
        'name': e.name,
        'content': e.content
    }
    csv_upload_label.set_text(f"✔️ {e.name}")
    ui.notify(f"Loaded '{e.name}' as the scan log.", type='positive')

def handle_cscan_upload(e: ui.events.UploadEventArguments):
    """Handle the upload of C-scan data files."""
    app_state['cscan_files'].append({
        'name': e.name,
        'content': e.content
    })
    cscan_upload_label.set_text(f"✔️ {len(app_state['cscan_files'])} C-Scan files loaded.")
    ui.notify(f"Added '{e.name}'. Total files: {len(app_state['cscan_files'])}", type='info')

def clear_cscan_files():
    """Clear the list of uploaded C-scan files."""
    app_state['cscan_files'] = []
    cscan_upload_label.set_text("C-Scan Data Files (.txt, .asc)")
    ui.notify("Cleared all uploaded C-Scan files.", type='warning')


async def build_composite_action():
    """Main action to build the composite C-scan."""
    if not app_state.get('csv_file') or not app_state.get('cscan_files'):
        ui.notify("Please upload both a CSV file and at least one C-Scan data file.", type='negative')
        return

    progress_dialog = ui.dialog()
    with progress_dialog:
        with ui.card().classes('glass-card'):
            ui.label('Building Composite...').classes('text-lg font-semibold')
            global progress_bar, progress_text
            progress_bar = ui.linear_progress(value=0, show_value=False)
            progress_text = ui.label('Initializing...')

    progress_dialog.open()
    
    temp_dir = None
    try:
        def update_progress(value, text):
            progress_bar.set_value(value)
            progress_text.set_text(text)
        
        await asyncio.sleep(0.1)
        update_progress(0.1, "Setting up temporary workspace...")

        # Create a temporary directory to store uploaded files
        temp_dir = tempfile.mkdtemp()
        
        # Save the CSV file
        csv_path = os.path.join(temp_dir, app_state['csv_file']['name'])
        with open(csv_path, 'wb') as f:
            f.write(app_state['csv_file']['content'].read())
            app_state['csv_file']['content'].seek(0) # Reset buffer

        # Save the C-scan files
        cscan_folder = os.path.join(temp_dir, 'cscan_data')
        os.makedirs(cscan_folder)
        for f_info in app_state['cscan_files']:
            with open(os.path.join(cscan_folder, f_info['name']), 'wb') as f:
                f.write(f_info['content'].read())
                f_info['content'].seek(0) # Reset buffer

        await asyncio.sleep(0.1)
        update_progress(0.3, "Building composite from scan data...")

        composite, vmin, vmax, mapped_files, missing, x_off, y_off = await asyncio.to_thread(
            build_composite,
            csv_path, cscan_folder, colourised_switch.value,
            area_filter_input.value, flip_scans_switch.value,
            scanning_correction_switch.value
        )
        
        update_progress(0.8, "Generating visualization...")
        await asyncio.sleep(0.1)
        
        app_state.update({
            'composite': composite, 'vmin': vmin, 'vmax': vmax,
            'x_offset': x_off, 'y_offset': y_off
        })

        min_thick = min_thickness_input.value or vmin
        max_thick = max_thickness_input.value or vmax
        
        plot_base64 = await asyncio.to_thread(
            plot_to_base64, composite, min_thick, max_thick, colourised_switch.value, x_off, y_off
        )
        
        update_progress(1.0, "Complete!")
        await asyncio.sleep(0.5)
        progress_dialog.close()
        
        result_image_display.set_source(f'data:image/png;base64,{plot_base64}')
        result_placeholder.set_visibility(False)
        result_image_display.set_visibility(True)

        log_text = (
            f"✅ Success! Mapped {len(mapped_files)}/{len(mapped_files) + len(missing)} scans.\n"
            f"📊 Data range: {vmin:.2f} - {vmax:.2f} mm\n"
            f"🖼 Canvas size: {composite.shape[1]} x {composite.shape[0]} px\n"
        )
        if missing:
            log_text += f"\n⚠️ Missing {len(missing)} scans:\n" + "\n".join(f"  • {s}" for s in missing[:5])
            if len(missing) > 5:
                log_text += "\n  • ..."
        
        log_area.set_value(log_text)
        ui.notify("Composite built successfully!", type='positive')

    except Exception as e:
        progress_dialog.close()
        log_area.set_value(f"❌ Error: {e}")
        ui.notify(f"An error occurred: {e}", type='negative', multi_line=True, classes='multi-line-notification')
        result_placeholder.set_visibility(True)
        result_image_display.set_visibility(False)
    finally:
        # Clean up the temporary directory
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


async def save_composite():
    """Save the current composite image to a file."""
    if app_state['composite'] is None:
        ui.notify("Please build a composite first.", type='warning')
        return
        
    ui.notify("Generating download... please wait.", type='ongoing')
    
    try:
        composite = app_state['composite']
        min_val = min_thickness_input.value or app_state['vmin']
        max_val = max_thickness_input.value or app_state['vmax']
        
        fig, ax = plt.subplots(figsize=(16, 9))
        cmap = "gray"
        if colourised_switch.value:
            colors = [(1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1)]
            positions = [0.0, 0.33, 0.66, 1.0]
            cmap = LinearSegmentedColormap.from_list("custom_cmap", list(zip(positions, colors)))
        
        height, width = composite.shape
        extent = [-app_state['x_offset'], width - app_state['x_offset'], -app_state['y_offset'], height - app_state['y_offset']]
        im = ax.imshow(composite, cmap=cmap, origin='lower', vmin=min_val, vmax=max_val, aspect='equal', extent=extent)
        
        plt.colorbar(im, label="Thickness (mm)")
        plt.title("Composite C-Scan Heatmap")
        plt.xlabel("X (mm)")
        plt.ylabel("Y (mm)")
        plt.tight_layout()
        
        buffer = BytesIO()
        await asyncio.to_thread(plt.savefig, buffer, format='png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        buffer.seek(0)
        
        # Use ui.download to send the file to the user
        ui.download(buffer.getvalue(), 'composite_c-scan.png', 'image/png')
        
    except Exception as e:
        ui.notify(f"Error creating download: {e}", type='negative')

async def process_single_scan_action(e: ui.events.UploadEventArguments):
    """Select and process a single C-scan file from an upload."""
    if not e.content:
        ui.notify("No file selected or file is empty.", type='warning')
        return

    progress_dialog = ui.dialog()
    with progress_dialog:
        with ui.card().classes('glass-card'):
            ui.label('Processing Single Scan...').classes('text-lg font-semibold')
            ui.spinner(size='lg')

    progress_dialog.open()

    try:
        # Run processing in a thread
        def process_and_plot():
            mat = parse_matrix_from_file(e.content)
            if mat.size == 0 or np.isnan(mat).all():
                raise ValueError("Scan file contains no valid data.")

            vmin, vmax = np.nanmin(mat), np.nanmax(mat)
            
            # For a single scan, offsets are 0
            x_offset, y_offset = 0, 0

            # Update app_state so "Save As Image" can work
            app_state.update({
                'composite': mat,
                'vmin': vmin,
                'vmax': vmax,
                'x_offset': x_offset,
                'y_offset': y_offset
            })
            
            min_thick = min_thickness_input.value or vmin
            max_thick = max_thickness_input.value or vmax
            colourised = colourised_switch.value

            plot_base64 = plot_to_base64(mat, min_thick, max_thick, colourised, x_offset, y_offset)
            return plot_base64, mat, vmin, vmax

        plot_base64, mat, vmin, vmax = await asyncio.to_thread(process_and_plot)

        progress_dialog.close()

        # Update UI
        result_image_display.set_source(f'data:image/png;base64,{plot_base64}')
        result_placeholder.set_visibility(False)
        result_image_display.set_visibility(True)

        log_text = (
            f"✅ Success! Loaded single scan: {e.name}\n"
            f"📊 Data range: {vmin:.2f} - {vmax:.2f} mm\n"
            f"🖼 Canvas size: {mat.shape[1]} x {mat.shape[0]} px\n"
        )
        log_area.set_value(log_text)
        ui.notify("Single scan processed successfully!", type='positive')

    except Exception as ex:
        progress_dialog.close()
        log_area.set_value(f"❌ Error: {ex}")
        ui.notify(f"An error occurred: {ex}", type='negative', multi_line=True, classes='multi-line-notification')
        result_placeholder.set_visibility(True)
        result_image_display.set_visibility(False)


# --- UI Definition ---
@ui.page('/')
def main_page():
    # This is the KEY FIX:
    # We target the main page container created by NiceGUI and remove its
    # max-width constraint, allowing our content to fill the entire width.
    ui.query('main.q-page').classes('w-full max-w-full p-0 m-0')

    # Set the dark mode and base theme colors
    ui.dark_mode(True)
    ui.colors(primary='#00ff88', secondary='#00bcd4', accent='#10b981', positive='#10b981')
    
    # Minimal, targeted styles for the aesthetic
    ui.add_head_html('''
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
        
        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(-45deg, #000, #0a0a0a, #1a1a1a, #001100, #001a1a, #002626, #1a1a1a, #000);
            background-size: 400% 400%;
            animation: darkGradientShift 30s ease-in-out infinite;
        }

        @keyframes darkGradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        .glass-card {
            background: rgba(10, 20, 20, 0.6);
            backdrop-filter: blur(15px);
            border: 1px solid rgba(0, 255, 136, 0.2);
            border-radius: 16px;
        }

        .neon-button {
            background: linear-gradient(145deg, #00ff88, #00bcd4);
            color: #000 !important;
            font-weight: 600;
        }
        
        .multi-line-notification {
            white-space: pre-wrap;
        }

        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: rgba(0,0,0,0.2); }
        ::-webkit-scrollbar-thumb { background: rgba(0, 255, 136, 0.5); border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: rgba(0, 255, 136, 0.8); }
    </style>
    ''')
    
    last_csv, last_cscan, last_min, last_max = load_settings()

    # The main layout is a full-screen row that doesn't wrap.
    with ui.row().classes('w-full h-screen p-0 m-0 no-wrap'):
        
        # --- LEFT SIDEBAR (FIXED WIDTH) ---
        with ui.column().classes('w-[450px] min-w-[450px] h-full p-6 space-y-6 overflow-y-auto flex-shrink-0'):
            ui.label('C-Scan Composite Analysis').classes('text-3xl font-bold text-white')
            ui.badge('v3.0-stable', color='positive').classes('w-fit')

            # Input Configuration Card
            with ui.card().classes('glass-card w-full'):
                with ui.card_section():
                    ui.label('Input Configuration').classes('text-lg font-semibold')
                with ui.card_section().classes('space-y-4'):
                    global csv_upload_label, cscan_upload_label
                    
                    with ui.upload(label='Scan Log CSV File', on_upload=handle_csv_upload, auto_upload=True).props('dark').classes('w-full') as csv_uploader:
                        csv_uploader.props('accept=".csv"')
                        csv_upload_label = ui.label('Scan Log CSV File (.csv)').classes('text-xs text-gray-400')

                    with ui.upload(label='C-Scan Data Files', on_upload=handle_cscan_upload, auto_upload=True, multiple=True).props('dark').classes('w-full') as cscan_uploader:
                        cscan_uploader.props('accept=".txt,.asc"')
                        cscan_upload_label = ui.label('C-Scan Data Files (.txt, .asc)').classes('text-xs text-gray-400')
                    
                    ui.button('Clear Uploaded Scans', icon='delete_sweep', on_click=clear_cscan_files).props('flat color=red-4').classes('w-full -mt-2')


            # Analysis Settings Card
            with ui.card().classes('glass-card w-full'):
                with ui.card_section():
                    ui.label('Analysis Settings').classes('text-lg font-semibold')
                with ui.card_section().classes('space-y-2'):
                    global colourised_switch, flip_scans_switch, scanning_correction_switch
                    global min_thickness_input, max_thickness_input, area_filter_input
                    
                    colourised_switch = ui.switch('Color Visualization', value=True)
                    flip_scans_switch = ui.switch('Flip Scans Vertically', value=False)
                    scanning_correction_switch = ui.switch('Correct Scanning Direction', value=True)
                    
                    with ui.row().classes('w-full gap-4'):
                        min_thickness_input = ui.number('Min Thickness', value=last_min, format='%.2f').props('dark').classes('w-full')
                        max_thickness_input = ui.number('Max Thickness', value=last_max, format='%.2f').props('dark').classes('w-full')
                    
                    area_filter_input = ui.input('Area Filter (e.g., strake 1)').props('dark').classes('w-full')
            
            # Operations Card
            with ui.card().classes('glass-card w-full'):
                with ui.card_section().classes('space-y-3'):
                    ui.button('Build Composite', icon='build', on_click=build_composite_action).classes('w-full neon-button h-12')
                    ui.button('Download Image', icon='download', on_click=save_composite).classes('w-full')
                    with ui.upload(label='Process Single Scan', on_upload=process_single_scan_action, auto_upload=True).props('dark').classes('w-full'):
                         ui.button('Process Single Scan', icon='image_search').classes('w-full')

        # --- RIGHT CONTENT AREA (FLEXIBLE WIDTH) ---
        with ui.column().classes('flex-grow h-full p-6 space-y-6 overflow-y-auto'):
            
            # Results Visualization Card
            with ui.card().classes('glass-card w-full h-3/5'):
                with ui.card_section():
                    ui.label('Composite Visualization').classes('text-lg font-semibold')
                with ui.card_section().classes('w-full h-full'):
                    global result_placeholder, result_image_display
                    with ui.column().classes('w-full h-full justify-center items-center text-center') as result_placeholder:
                        ui.icon('analytics', size='6rem').classes('text-gray-500')
                        ui.label('Build a composite to see results here').classes('text-gray-400 font-semibold')
                    
                    result_image_display = ui.image().classes('w-full h-full object-contain rounded-lg').set_visibility(False)

            # Process Log Card
            with ui.card().classes('glass-card w-full flex-grow'):
                with ui.card_section():
                    ui.label('Process Log').classes('text-lg font-semibold')
                with ui.card_section().classes('w-full h-full'):
                    global log_area
                    log_area = ui.textarea().props('dark readonly outlined').classes('w-full h-full font-mono text-xs')


# --- Application Entry Point ---
if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        title='C-Scan Composite Analysis',
        favicon='🔬',
        dark=True,
        # The host and port will be set by the cloud provider,
        # but these are good defaults for local testing.
        host='0.0.0.0',
        port=8080,
        reload=False
    )
