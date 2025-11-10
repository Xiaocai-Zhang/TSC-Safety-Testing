from __future__ import absolute_import, print_function
from pywinauto import application, timings, Desktop
from alive_progress import alive_bar
import os, time
import psutil




Num_of_simulation = 1
debug_mode = False
hide_window = False

# thresholds
maxTTC = 5          # s
maxPET = 4          # s
Rear_end_angle = 30 # deg
crossing_angle = 85 # deg

max_timeout = 50
max_analyse_timeout = 3600
access_interval = 0.2
max_attempts = 20

SSAM_exe_path = r'SSAM\SSAM3EXE\SSAM3.exe'
SSAM_exe = os.path.join(os.getcwd(), SSAM_exe_path)


def kill_stale_ssam():
    for p in psutil.process_iter(['name']):
        try:
            nm = p.info['name']
            if nm and nm.lower().startswith('ssam3'):
                p.kill()
        except Exception:
            pass

def dismiss_any_dialog(app, main_handle, timeout=10):
    t0 = time.time()
    while time.time() - t0 < timeout:
        for w in Desktop(backend="win32").windows():
            try:
                if w.handle != main_handle and w.class_name() in ("#32770", "Dialog"):
                    for btn_text in ["&Yes", "Yes", "&OK", "OK", "&Overwrite", "Overwrite", "&Continue", "Continue", "&确定", "确定"]:
                        try:
                            w[btn_text].click()
                            time.sleep(0.2)
                            return True
                        except Exception:
                            pass
                    try:
                        w.type_keys("{ENTER}")
                        return True
                    except Exception:
                        pass
            except Exception:
                pass
        time.sleep(0.2)
    return False

def wait_analysis_complete(window, analyse_timeout=3600, poll=0.5):
    t0 = time.time()
    last = ""
    while time.time() - t0 < analyse_timeout:
        try:
            txts = window.Static7.texts()
            cur = txts[0] if txts else ""
            if "Analysis is completed." in cur:
                return True
            if cur != last:
                last = cur
        except Exception:
            pass
        time.sleep(poll)
    return False

def wait_file_ready(path, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if os.path.exists(path):
            try:
                with open(path, 'rb'):
                    return True
            except Exception:
                pass
        time.sleep(0.2)
    return False

def Modify_Value(Edit_key, val):
    Edit_key.set_text(val)

def SSAM_Setting_Check(window, trj_path):
    if str(maxTTC) != window.Edit.window_text():
        raise ValueError("TTC")
    if str(maxPET) != window.Edit2.window_text():
        raise ValueError("PET")
    if str(Rear_end_angle) != window.Edit3.window_text():
        raise ValueError("Rear end Angle")
    if str(crossing_angle) != window.Edit4.window_text():
        raise ValueError("Crossing Angle")
    if trj_path != window.ListBox.item_texts()[0]:
        raise ValueError("path")

def Get_Path(trj_basename, EvalOutDir):
    trj_ID = os.path.splitext(trj_basename)[0]
    conflict_data_path = os.path.abspath(os.path.join(EvalOutDir, trj_ID + '.csv'))
    summary_path = os.path.abspath(os.path.join(EvalOutDir, trj_ID + '_summary.csv'))

    conflict_data_path = conflict_data_path.replace('/', '\\')
    summary_path = summary_path.replace('/', '\\')

    if os.path.exists(conflict_data_path): os.remove(conflict_data_path)
    if os.path.exists(summary_path): os.remove(summary_path)

    return trj_ID, conflict_data_path, summary_path


def Start(EvalOutDir):
    os.makedirs(EvalOutDir, exist_ok=True)
    # 1) collect .trj once
    trj_files = [f for f in os.listdir(EvalOutDir) if f.lower().endswith(".trj")]
    global Num_of_simulation
    Num_of_simulation = len(trj_files)

    with alive_bar(Num_of_simulation, spinner='classic') as bar:
        for file in trj_files:
            trj_path = os.path.abspath(os.path.join(EvalOutDir, file))
            trj_ID, conflict_data_path, summary_path = Get_Path(file, EvalOutDir)

            # 2) skip if outputs ready
            if all(map(os.path.exists, [summary_path, conflict_data_path])):
                try:
                    with open(summary_path,'rb'), open(conflict_data_path,'rb'):
                        bar()
                        continue
                except Exception:
                    pass

            app = application.Application(backend="win32")
            try:
                # --- Launch (unchanged, but no minimize/restore) ---
                for i in range(max_attempts):
                    try:
                        app.start(SSAM_exe)
                        window = app['SSAM3']
                        main_handle = window.element_info.handle
                        window = app.window(title_re="SSAM3", handle=main_handle)
                        window.wait('exists', max_timeout, access_interval)
                        break
                    except timings.TimeoutError:
                        if i == max_attempts - 1:
                            raise RuntimeError("reach max attempts")
                        print("Retry: open SSAM3")

                # --- Open file (resolve controls once) ---
                for i in range(max_attempts):
                    try:
                        window.Add.click()
                        open_dlg = app['Open']
                        open_dlg.wait('exists', max_timeout, access_interval)

                        if window.ListBox.item_count() == 0:
                            open_dlg.Edit.set_text(trj_path)
                        open_dlg[u'&Open'].click()
                        timings.wait_until(max_timeout, access_interval, lambda: open_dlg.exists(), False)

                        timings.wait_until(max_timeout, access_interval, lambda: window.ListBox.item_count(), 1)
                        timings.wait_until(max_timeout, access_interval, lambda: window.ListBox.item_texts()[0], trj_path)
                        dismiss_any_dialog(app, main_handle, timeout=2)
                        break
                    except timings.TimeoutError:
                        if i == max_attempts - 1:
                            raise RuntimeError("reach max attempts (open)")
                        print("Retry: add trj file to list")

                # --- Set params (same) ---
                Modify_Value(window.Edit,  maxTTC)
                Modify_Value(window.Edit2, maxPET)
                Modify_Value(window.Edit3, Rear_end_angle)
                Modify_Value(window.Edit4, crossing_angle)
                SSAM_Setting_Check(window, trj_path)

                # --- Analyze (keep robust wait; remove CPU idle wait) ---
                for i in range(max_attempts):
                    try:
                        window.Analyze.click()
                        if not wait_analysis_complete(window, analyse_timeout=max_analyse_timeout, poll=access_interval):
                            raise timings.TimeoutError("analysis timeout")
                        dismiss_any_dialog(app, main_handle, timeout=2)
                        break
                    except timings.TimeoutError:
                        if i == max_attempts - 1:
                            raise RuntimeError("reach max attempts (analyze)")
                        print("Retry: analyse trj data")

                tabc = window.TabControl

                # --- Export Summary (resolve controls once, no global dialog scans) ---
                for i in range(max_attempts):
                    try:
                        tabc.select(2)
                        window.Button2.click()
                        save_dlg = app['Save As']
                        save_dlg.wait('exists', max_timeout, access_interval)
                        save_dlg.Edit.set_text(summary_path)
                        save_dlg[u'&Save'].click()
                        timings.wait_until(max_timeout, access_interval, lambda: save_dlg.exists(), False)
                        if not wait_file_ready(summary_path, timeout=30):
                            raise RuntimeError("summary not ready")
                        dismiss_any_dialog(app, main_handle, timeout=2)
                        break
                    except Exception:
                        if i == max_attempts - 1:
                            raise
                        print("Retry: save summary")

                # --- Export Conflict ---
                for i in range(max_attempts):
                    try:
                        try:
                            tabc.select(1)
                        except RuntimeError:
                            window.wait('exists enabled visible ready', max_timeout, access_interval)
                        window.Button2.click()
                        save_dlg = app['Save As']
                        save_dlg.wait('exists', max_timeout, access_interval)
                        save_dlg.Edit.set_text(conflict_data_path)
                        save_dlg[u'&Save'].click()
                        timings.wait_until(max_timeout, access_interval, lambda: save_dlg.exists(), False)
                        if not wait_file_ready(conflict_data_path, timeout=30):
                            raise RuntimeError("conflict not ready")
                        dismiss_any_dialog(app, main_handle, timeout=2)
                        break
                    except Exception:
                        if i == max_attempts - 1:
                            raise
                        print("Retry: save conflict")

            finally:
                try: app.kill()
                except: pass
                time.sleep(0.3)   # shorter cool-down
            bar()
