# --- Core Libraries ---
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from scipy.stats import ttest_1samp
import bctpy.bct as bct
from scipy.stats import pearsonr
from matplotlib.ticker import PercentFormatter
import gc

# --- MNE-Python Ecosystem ---
import mne
from mne_bids import BIDSPath, read_raw_bids, get_entity_vals
from mne.preprocessing import ICA
from mne_connectivity import spectral_connectivity_epochs
from mne_connectivity.viz import plot_connectivity_circle
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs

# --- Graph Theory (using bctpy) ---
import bct
import mne.viz
import networkx as nx

# --- Machine Learning & Decoding ---
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, ShuffleSplit
from mne.decoding import CSP
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# --- Utility ---
from matplotlib.colors import TwoSlopeNorm
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ====================================================================
# --- I. CONFIGURATION AND GLOBAL SETUP ---
# ====================================================================

# The user-provided path to the downloaded dataset
# !!! ADJUST THIS PATH !!!
BIDS_ROOT = r"C:\Users\mypc\OneDrive\Documents\BCI_Research"
# Define a path for processed data (derivatives)
DERIVATIVES_ROOT = os.path.join(BIDS_ROOT, 'derivatives', 'mne-pipeline')
FIGURES_ROOT = os.path.join(DERIVATIVES_ROOT, 'figures')
# Ensure directories exist
os.makedirs(DERIVATIVES_ROOT, exist_ok=True)
os.makedirs(FIGURES_ROOT, exist_ok=True)

try:
    SUBJECTS_LIST = get_entity_vals(BIDS_ROOT, 'subject')
except:
    print(f"!! ERROR: Could not find subjects in BIDS_ROOT: {BIDS_ROOT}")
    SUBJECTS_LIST = []
    
print(f"Found {len(SUBJECTS_LIST)} subjects.")

# --- Global Analysis Parameters ---
DOWNSAMPLE_FREQ = 125  # Hz (Original is 250 Hz )
LINE_NOISE_FREQ = 60   # Hz (Dataset from Colombia , which uses 60Hz )
EPOCH_TMIN = -1.0
EPOCH_TMAX = 4.0
BASELINE_TIMING = (-1.0, 0.0)
ICA_N_COMPONENTS = 15
ICA_RANDOM_STATE = 97
REJECT_CRITERIA = dict(eeg=150e-6) # 150 µV

FREQUENCY_BANDS = {
    'mu': (8, 13),
    'beta': (13, 30),
    'broadband': (1, 40)
}

EVENT_ID_MAP = {
    '1': 1,  # Sit-to-stand imagery
    '2': 2,     # Sitting idle
    '3': 3,  # Stand-to-sit imagery
    '4': 4      # Standing idle
}

# Global dictionary to store and pass results between functions
# This is a simple way to manage state across function calls
RESULTS_CACHE = {
    'all_epochs_data': {}, # Stores MNE Epochs objects
    'avg_beta_plv': None,  # Stores Group Beta PLV Matrix
    'avg_beta_coh': None,  # Stores Group Beta Coherence Matrix
    'ch_names': [],        # Channel names
    'sfreq': None          # Sampling frequency
}

# --- Utility Function to Load Epochs ---
def load_epochs(subject_list):
    """Loads cleaned epochs from .fif files if they exist."""
    print("Loading pre-processed epochs...")
    all_epochs_data = {}
    ch_names = []
    sfreq = None
    
    for subject in subject_list:
        subject_dir = os.path.join(DERIVATIVES_ROOT, f'sub-{subject}')
        out_fname = os.path.join(
            subject_dir, f'sub-{subject}_task-sitstand_proc-clean-epo.fif'
        )
        if os.path.exists(out_fname):
            try:
                epochs = mne.read_epochs(out_fname, preload=True, verbose=False)
                all_epochs_data[subject] = epochs
                if not ch_names:
                    ch_names = epochs.info['ch_names']
                    sfreq = epochs.info['sfreq']
            except Exception as e:
                print(f"! Failed to load epochs for {subject}: {e}")
        else:
            print(f"! Missing epochs file for {subject}: {out_fname}")
    
    if all_epochs_data:
        RESULTS_CACHE['all_epochs_data'] = all_epochs_data
        RESULTS_CACHE['ch_names'] = ch_names
        RESULTS_CACHE['sfreq'] = sfreq
        print(f"Successfully loaded epochs for {len(all_epochs_data)} subjects.")
        return True
    return False

# ====================================================================
# --- II. ANALYSIS FUNCTIONS (MODULARIZED) ---
# ====================================================================

def run_preprocessing(subject_list):
    print("\n--- Starting Part 1: Preprocessing Pipeline (Time-consuming) ---")
    all_epochs_data = {}
    all_epoch_fnames = []
    
    # [... Your existing preprocessing loop goes here ...]
    # Keep the logic exactly as it is in your uploaded script,
    # ensuring the final 'epochs' object is correctly saved and
    # added to all_epochs_data.
    
    for subject in subject_list:
        print(f"\nProcessing Subject: {subject}")
        try:
            # --- Step 1.1: Data Import (Metadata, Events, Channels) ---
            bids_path = BIDSPath(subject=subject, task='sitstand', datatype='eeg', root=BIDS_ROOT)
            raw = read_raw_bids(bids_path, verbose=False).load_data()
            montage = mne.channels.make_standard_montage('standard_1020')
            raw.set_montage(montage, on_missing='ignore')
            
            # --- Step 1.2-1.6 (Downsampling, Re-referencing, Filtering, ICA, Interpolation) ---
            raw.resample(DOWNSAMPLE_FREQ)
            raw.set_eeg_reference('average', projection=True)
            raw.notch_filter(freqs=LINE_NOISE_FREQ)
            raw.filter(l_freq=FREQUENCY_BANDS['broadband'][0], h_freq=FREQUENCY_BANDS['broadband'][1])
            
            ica_raw = raw.copy().filter(l_freq=1.0, h_freq=None)
            ica = ICA(n_components=ICA_N_COMPONENTS, random_state=ICA_RANDOM_STATE, max_iter='auto')
            ica.fit(ica_raw, reject_by_annotation=True)
            ica.exclude = [0, 1, 6, 7, 9, 10] # Manual Exclusion
            ica.apply(raw)
            print(f"  > ICA: Excluded {len(ica.exclude)} components.")
            
            raw.interpolate_bads(reset_bads=False, mode='accurate')

            # --- Step 1.7-1.9 (Epoching, Baseline, Rejection, Saving) ---
            events, _ = mne.events_from_annotations(raw, event_id=EVENT_ID_MAP)
            
            epochs = mne.Epochs(
                raw, events, event_id=EVENT_ID_MAP, 
                tmin=EPOCH_TMIN, tmax=EPOCH_TMAX, 
                baseline=BASELINE_TIMING, reject=REJECT_CRITERIA, 
                preload=True, reject_by_annotation=True
            )
            
            print(f"  > Epochs: Created {len(epochs)} epochs.")
            
            subject_dir = os.path.join(DERIVATIVES_ROOT, f'sub-{subject}')
            os.makedirs(subject_dir, exist_ok=True)
            out_fname = os.path.join(
                subject_dir, f'sub-{subject}_task-sitstand_proc-clean-epo.fif'
            )
            epochs.save(out_fname, overwrite=True)
            all_epoch_fnames.append(out_fname)
            all_epochs_data[subject] = epochs
            print(f"  > Saved cleaned epochs to: {out_fname}")

        except Exception as e:
            print(f"! FAILED to process {subject}. Error: {e}")

    print("\n--- Preprocessing Complete ---")
    if all_epochs_data:
        first_subject_key = list(all_epochs_data.keys())[0]
        RESULTS_CACHE['ch_names'] = all_epochs_data[first_subject_key].info['ch_names']
        RESULTS_CACHE['sfreq'] = all_epochs_data[first_subject_key].info['sfreq']
        RESULTS_CACHE['all_epochs_data'] = all_epochs_data
        return True
    return False



def run_connectivity(all_epochs_data, ch_names, sfreq):
    print("\n[Analysis 2.1] Computing PLV and Coherence (Group & Individual)...")
    
    connectivity_methods = ['plv', 'coh']
    all_connectivity_results = {'mu':[], 'beta':[]}
    all_subject_indices = []
    all_global_efficiencies = []
    
    for subject, epochs in all_epochs_data.items():
        print(f"  > Processing connectivity for Subject: {subject}")
        
        # --- SCIENTIFIC CORRECTION: Analyze MI epochs, not all epochs ---
        try:
            epochs_mi = mne.epochs.concatenate_epochs([epochs['1'], epochs['3']])
        except KeyError:
            print(f" ! Skipping Subject {subject}: Missing event '1' or '3'.")
            continue
        except ValueError:
            print(f" ! Skipping Subject {subject}: No data for event '1' or '3'.")
            continue
        # -----------------------------------------------------------

        # Mu Band (8-13 Hz) 
        con_mu = spectral_connectivity_epochs(
            epochs_mi, method=connectivity_methods,
            fmin=FREQUENCY_BANDS['mu'][0], fmax=FREQUENCY_BANDS['mu'][1],
            faverage=True, sfreq=sfreq, n_jobs=-1
        )
        # Beta Band (13-30 Hz) 
        con_beta = spectral_connectivity_epochs(
            epochs_mi, method=connectivity_methods,
            fmin=FREQUENCY_BANDS['beta'][0], fmax=FREQUENCY_BANDS['beta'][1],
            faverage=True, sfreq=sfreq, n_jobs=-1
        )
        all_connectivity_results['mu'].append(con_mu)
        all_connectivity_results['beta'].append(con_beta)

        # --- Calculate and Save Individual Graph Metrics ---
        try:
            # We focus on Beta PLV (index [0])
            con_beta_plv_data = con_beta[0].get_data(output='dense').squeeze()
            
            # Symmetrize matrix for undirected graph metrics
            con_beta_plv_data = (con_beta_plv_data + con_beta_plv_data.T) / 2
            
            # Call the function to calculate and save metrics for this subject
            global_eff = calculate_and_save_individual_graph_metrics(
                con_matrix=con_beta_plv_data,
                ch_names=ch_names,
                subject_id=subject,
                metric_name='Beta_PLV',
                output_dir=DERIVATIVES_ROOT 
            )
            
            # Store the results for the final correlation
            if not np.isnan(global_eff):
                all_subject_indices.append(int(subject))
                all_global_efficiencies.append(global_eff)
            else:
                print(f"  > Skipping subject {subject} from correlation due to graph metric error.")
                
        except Exception as e:
            print(f"! FAILED to calculate individual graph metrics for {subject}: {e}")
        # --- END OF ADDED BLOCK ---


    try:
        # --- Group Averaging ---
        if not all_connectivity_results['beta'] or not all_connectivity_results['mu']:
            print("! FAILED: No connectivity results were computed. Skipping group averaging.")
            n_ch = len(ch_names)
            avg_beta_plv = np.full((n_ch, n_ch), np.nan)
            avg_beta_coh = np.full((n_ch, n_ch), np.nan)
            avg_mu_plv = np.full((n_ch, n_ch), np.nan)
            avg_mu_coh = np.full((n_ch, n_ch), np.nan)
        else:
            # We have data, proceed with averaging
            # Beta PLV (access the 1st item, index [0])
            avg_beta_plv = np.mean([res[0].get_data(output='dense')[:, :, 0] for res in all_connectivity_results['beta']], axis=0)
            np.save(os.path.join(DERIVATIVES_ROOT, 'group_connect_beta_plv.npy'), avg_beta_plv)
            
            # Beta Coherence (access the 2nd item, index [1])
            avg_beta_coh = np.mean([res[1].get_data(output='dense')[:, :, 0] for res in all_connectivity_results['beta']], axis=0)
            np.save(os.path.join(DERIVATIVES_ROOT, 'group_connect_beta_coh.npy'), avg_beta_coh)

            # Mu PLV (access the 1st item, index [0])
            avg_mu_plv = np.mean([res[0].get_data(output='dense')[:, :, 0] for res in all_connectivity_results['mu']], axis=0)
            np.save(os.path.join(DERIVATIVES_ROOT, 'group_connect_mu_plv.npy'), avg_mu_plv)

            # Mu Coherence (access the 2nd item, index [1])
            avg_mu_coh = np.mean([res[1].get_data(output='dense')[:, :, 0] for res in all_connectivity_results['mu']], axis=0)
            np.save(os.path.join(DERIVATIVES_ROOT, 'group_connect_mu_coh.npy'), avg_mu_coh)

            print("  > PLV/Coherence group average computation complete.")
            
            # --- Plotting (This part is unchanged) ---
            n_channels = len(ch_names)
            
            # 1. Beta PLV Plot
            fig_plv, ax_plv = plt.subplots(figsize=(6, 5))
            im_plv = ax_plv.imshow(avg_beta_plv, cmap='viridis', origin='lower',
                             norm=TwoSlopeNorm(vmin=avg_beta_plv.min(), vcenter=np.mean(avg_beta_plv), vmax=avg_beta_plv.max()))
            ax_plv.set_xticks(np.arange(n_channels)); ax_plv.set_yticks(np.arange(n_channels))
            ax_plv.set_xticklabels(ch_names, rotation=90, fontsize=8); ax_plv.set_yticklabels(ch_names, fontsize=8)
            ax_plv.set_title("Group-Average Beta Band (13-30 Hz) PLV")
            fig_plv.colorbar(im_plv, ax=ax_plv, label="Phase-Locking Value (PLV)")
            plt.tight_layout()
            fig_plv.savefig(os.path.join(FIGURES_ROOT, 'group_connect_beta_plv.png'), dpi=300)
            plt.show(block=False)
            
            # 2. Beta Coherence Plot
            fig_coh, ax_coh = plt.subplots(figsize=(6, 5))
            im_coh = ax_coh.imshow(avg_beta_coh, cmap='viridis', origin='lower',
                             norm=TwoSlopeNorm(vmin=avg_beta_coh.min(), vcenter=np.mean(avg_beta_coh), vmax=avg_beta_coh.max()))
            ax_coh.set_xticks(np.arange(n_channels)); ax_coh.set_yticks(np.arange(n_channels))
            ax_coh.set_xticklabels(ch_names, rotation=90, fontsize=8); ax_coh.set_yticklabels(ch_names, fontsize=8)
            ax_coh.set_title("Group-Average Beta Band (13-30 Hz) Coherence")
            fig_coh.colorbar(im_coh, ax=ax_coh, label="Coherence")
            plt.tight_layout()
            fig_coh.savefig(os.path.join(FIGURES_ROOT, 'group_connect_beta_coh.png'), dpi=300)
            plt.show(block=False)

            #... Include Mu plots here as well...

        # --- ADD THIS BLOCK TO SAVE INDIVIDUAL GLOBAL METRICS ---
        if all_subject_indices:
            global_metrics_df = pd.DataFrame({
                'subject_index': all_subject_indices,
                'Beta_PLV_Global_Efficiency': all_global_efficiencies
            })
            gmetrics_path = os.path.join(DERIVATIVES_ROOT, 'individual_global_metrics.csv')
            global_metrics_df.to_csv(gmetrics_path, index=False)
            print(f"  > Saved individual global metrics to: {gmetrics_path}")
            RESULTS_CACHE['global_metrics_df'] = global_metrics_df
        
        RESULTS_CACHE['avg_beta_plv'] = avg_beta_plv
        RESULTS_CACHE['avg_beta_coh'] = avg_beta_coh
        RESULTS_CACHE['avg_mu_plv'] = avg_mu_plv 
        
        return {'avg_beta_plv': avg_beta_plv, 'avg_beta_coh': avg_beta_coh, 'avg_mu_plv': avg_mu_plv}

    except Exception as e:
        print("\n" + "!"*60)
        print("! --- CRITICAL ERROR IN run_connectivity (Post-Loop) --- !")
        print(f"! An unhandled exception occurred: {e}")
        import traceback
        traceback.print_exc()
        print("! Function is returning None, which causes the main menu crash.")
        print("!"*60 + "\n")
        return None # Explicitly return None



def calculate_and_save_individual_graph_metrics(con_matrix, ch_names, subject_id, metric_name, output_dir):
    """
    Calculates and saves NODAL graph metrics for a single subject.
    This is called from within the run_connectivity loop.
    Returns the Global Efficiency for this subject.
    """
    try:
        A = con_matrix
        
        # 1. Calculate Distances for Path Length
        with np.errstate(divide='ignore'):
            A_dist = 1. / A
        A_dist[np.isinf(A_dist)] = 0
        np.fill_diagonal(A_dist, 0)

        # 2. Global Metrics
        # We only need Global Efficiency for the correlation analysis
        char_path, global_eff, _, _, _ = bct.charpath(A_dist)
        # Handle potential infinite path length if graph is disconnected
        if np.isinf(global_eff):
            global_eff = 0
        
        # 3. Nodal Metrics (for correlation)
        nodal_strength = bct.strengths_und(A) 
        nodal_centrality = bct.betweenness_wei(A_dist)
        
        # 4. Create and Save DataFrame
        graph_metrics_df = pd.DataFrame({
            'Channel': ch_names,
            'Nodal_Strength': nodal_strength,
            'Betweenness_Centrality': nodal_centrality
        })
        
        # Create a unique filename for this subject and metric
        subject_dir = os.path.join(output_dir, f'sub-{subject_id}')
        os.makedirs(subject_dir, exist_ok=True) # Ensure subject dir exists
        
        # --- FIX: Save to subject dir, not main derivatives dir ---
        csv_path_graph = os.path.join(
            subject_dir, 
            f'sub-{subject_id}_desc-{metric_name}_nodal_metrics.csv'
        )
        graph_metrics_df.to_csv(csv_path_graph, index=False)
        
        # Return the global efficiency to be collected
        return global_eff

    except Exception as e:
        print(f"  ! Error in individual graph metrics for {subject_id}: {e}")
        return np.nan # Return NaN if it fails

# --- Group Averaging ---
    # Check if we have any results to average
    if not all_connectivity_results['beta'] or not all_connectivity_results['mu']:
        print("! FAILED: No connectivity results were computed. Skipping group averaging.")
        # Return empty/NaN matrices to prevent crashing
        n_ch = len(ch_names)
        avg_beta_plv = np.full((n_ch, n_ch), np.nan)
        avg_beta_coh = np.full((n_ch, n_ch), np.nan)
        avg_mu_plv = np.full((n_ch, n_ch), np.nan)
        avg_mu_coh = np.full((n_ch, n_ch), np.nan)
    else:
        # We have data, proceed with averaging
        # Beta PLV
        avg_beta_plv = np.mean([res[0].get_data(output='dense')[:, :, 0] for res in all_connectivity_results['beta']], axis=0)
        np.save(os.path.join(DERIVATIVES_ROOT, 'group_connect_beta_plv.npy'), avg_beta_plv)
        
        # Beta Coherence
        avg_beta_coh = np.mean([res[1].get_data(output='dense')[:, :, 0] for res in all_connectivity_results['beta']], axis=0)
        np.save(os.path.join(DERIVATIVES_ROOT, 'group_connect_beta_coh.npy'), avg_beta_coh)

        # Mu PLV
        avg_mu_plv = np.mean([res[0].get_data(output='dense')[:, :, 0] for res in all_connectivity_results['mu']], axis=0)
        np.save(os.path.join(DERIVATIVES_ROOT, 'group_connect_mu_plv.npy'), avg_mu_plv)

        # Mu Coherence
        avg_mu_coh = np.mean([res[1].get_data(output='dense')[:, :, 0] for res in all_connectivity_results['mu']], axis=0)
        np.save(os.path.join(DERIVATIVES_ROOT, 'group_connect_mu_coh.npy'), avg_mu_coh)

        print("  > PLV/Coherence group average computation complete.")

    # --- ADD THIS BLOCK TO SAVE INDIVIDUAL GLOBAL METRICS ---
    if all_subject_indices:
        global_metrics_df = pd.DataFrame({
            'subject_index': all_subject_indices,
            'Beta_PLV_Global_Efficiency': all_global_efficiencies
        })
        gmetrics_path = os.path.join(DERIVATIVES_ROOT, 'individual_global_metrics.csv')
        global_metrics_df.to_csv(gmetrics_path, index=False)
        print(f"  > Saved individual global metrics to: {gmetrics_path}")
        RESULTS_CACHE['global_metrics_df'] = global_metrics_df


    n_channels = len(ch_names)
    
    # 1. Beta PLV Plot
    fig_plv, ax_plv = plt.subplots(figsize=(6, 5))
    im_plv = ax_plv.imshow(avg_beta_plv, cmap='viridis', origin='lower',
               norm=TwoSlopeNorm(vmin=avg_beta_plv.min(), vcenter=np.mean(avg_beta_plv), vmax=avg_beta_plv.max()))
    ax_plv.set_xticks(np.arange(n_channels)); ax_plv.set_yticks(np.arange(n_channels))
    ax_plv.set_xticklabels(ch_names, rotation=90, fontsize=8); ax_plv.set_yticklabels(ch_names, fontsize=8)
    ax_plv.set_title("Group-Average Beta Band (13-30 Hz) PLV")
    fig_plv.colorbar(im_plv, ax=ax_plv, label="Phase-Locking Value (PLV)")
    plt.tight_layout()
    fig_plv.savefig(os.path.join(FIGURES_ROOT, 'group_connect_beta_plv.png'), dpi=300)
    plt.show(block=False)
    
    # 2. Beta Coherence Plot
    fig_coh, ax_coh = plt.subplots(figsize=(6, 5))
    im_coh = ax_coh.imshow(avg_beta_coh, cmap='viridis', origin='lower',
                norm=TwoSlopeNorm(vmin=avg_beta_coh.min(), vcenter=np.mean(avg_beta_coh), vmax=avg_beta_coh.max()))
    ax_coh.set_xticks(np.arange(n_channels)); ax_coh.set_yticks(np.arange(n_channels))
    ax_coh.set_xticklabels(ch_names, rotation=90, fontsize=8); ax_coh.set_yticklabels(ch_names, fontsize=8)
    ax_coh.set_title("Group-Average Beta Band (13-30 Hz) Coherence")
    fig_coh.colorbar(im_coh, ax=ax_coh, label="Coherence")
    plt.tight_layout()
    fig_coh.savefig(os.path.join(FIGURES_ROOT, 'group_connect_beta_coh.png'), dpi=300)
    plt.show(block=False)

    #... Include Mu plots here as well...
    
    RESULTS_CACHE['avg_beta_plv'] = avg_beta_plv
    RESULTS_CACHE['avg_beta_coh'] = avg_beta_coh
    # Add avg_mu_plv to cache for step 10
    RESULTS_CACHE['avg_mu_plv'] = avg_mu_plv 
    return {'avg_beta_plv': avg_beta_plv, 'avg_beta_coh': avg_beta_coh, 'avg_mu_plv': avg_mu_plv}

def run_graph_theory(matrix, matrix_name, ch_names, summary_results):
    print(f"\n[Analysis 2.2] Computing Graph Theoretical Metrics ({matrix_name})...")
    
    A = matrix # Use the provided matrix
    
    # 1. Global Metrics
    with np.errstate(divide='ignore'):
        A_dist = 1. / A
    A_dist[np.isinf(A_dist)] = 0
    np.fill_diagonal(A_dist, 0)
    
    char_path, global_eff, _, _, _ = bct.charpath(A_dist)
    global_eff_wei = bct.efficiency_wei(A) 
 
    if np.isinf(global_eff):
        global_eff = 0
    
    # Add to summary results with a unique key
    summary_results[f'graph_char_path_{matrix_name}'] = [char_path]
    summary_results[f'graph_global_eff_{matrix_name}'] = [global_eff]
    summary_results[f'graph_global_eff_weighted_{matrix_name}'] = [global_eff_wei]
    
    print(f"  > Characteristic Path Length: {char_path:.4f}")
    
    # 2. Nodal Metrics
    nodal_strength = bct.strengths_und(A) 
    nodal_centrality = bct.betweenness_wei(A_dist)
    
    graph_metrics_df = pd.DataFrame({
        'Channel': ch_names,
        'Nodal Strength': nodal_strength,
        'Betweenness Centrality': nodal_centrality
    }).sort_values(by='Betweenness Centrality', ascending=False)
    
    # Save to a unique CSV
    csv_path_graph = os.path.join(DERIVATIVES_ROOT, f'group_graph_metrics_{matrix_name}.csv')
    graph_metrics_df.to_csv(csv_path_graph, index=False)
    print(f"  > Saved graph metrics to: {csv_path_graph}")
    print(graph_metrics_df.to_string())
    
    # --- PLOT: Nodal Graph Metrics ---
    
    # --- ROBUST FIX: Check if there is any data to plot ---
    if graph_metrics_df['Betweenness Centrality'].sum() == 0:
        print(f"  > Skipping centrality plot for {matrix_name}: All centrality values are zero.")
    else:
        df_sorted = graph_metrics_df.sort_values('Betweenness Centrality', ascending=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(df_sorted['Channel'], df_sorted['Betweenness Centrality'], color='skyblue')
        ax.set_xlabel('Betweenness Centrality')
        ax.set_title(f'Nodal Betweenness Centrality ({matrix_name} Network)') # Add unique title
        plt.tight_layout()
        # Save to a unique PNG
        fig.savefig(os.path.join(FIGURES_ROOT, f'group_graph_nodal_centrality_{matrix_name}.png'), dpi=300)
        plt.show(block=False)
    plot_nodal_strength(graph_metrics_df, matrix_name, ch_names)
    
    return summary_results

def plot_nodal_strength(graph_metrics_df, matrix_name, ch_names):
    """
    Plots and saves a horizontal bar chart for Nodal Strength.
    """
    print(f"  > Plotting Nodal Strength for {matrix_name}...")
    
    # --- FIX 1: Use the correct column name "Nodal Strength" ---
    df_to_plot = graph_metrics_df[graph_metrics_df["Nodal Strength"] > 0]
    
    if df_to_plot.empty:
        print(f"  > Skipping strength plot for {matrix_name}: All strength values are zero.")
        return

    # Sort ascending for a clean horizontal bar chart
    df_sorted = df_to_plot.sort_values('Nodal Strength', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # --- FIX 2: Pass the 'Nodal Strength' column for the bar widths ---
    ax.barh(df_sorted['Channel'], df_sorted['Nodal Strength'], color='tomato')
    
    ax.set_xlabel('Nodal Strength (Sum of PLV/Coherence)')
    ax.set_title(f'Nodal Strength ({matrix_name} Network)')
    plt.tight_layout()
    
    # Save to a unique PNG
    fig_filename = os.path.join(FIGURES_ROOT, f'group_graph_nodal_strength_{matrix_name}.png')
    fig.savefig(fig_filename, dpi=300)
    print(f"  > Saved Nodal Strength plot to: {fig_filename}")
    plt.show(block=False)


def run_granger_causality(all_epochs_data, ch_names, sfreq):
    print("\n[Analysis 2.3] Computing Granger Causality...")

    # Restricted to flow FROM C3, Cz, C4 to all other channels
    seed_channels = ['C3', 'Cz', 'C4']
    target_channels = [ch for ch in ch_names if ch not in seed_channels]
    seed_idx = [ch_names.index(ch) for ch in seed_channels if ch in ch_names]
    target_idx = [ch_names.index(ch) for ch in target_channels if ch in ch_names]

    # --- Setup indices for MNE-Connectivity ---
    gc_seeds_list_of_lists = []
    gc_targets_list_of_lists = []
    
    for s_index in seed_idx:
        for t_index in target_idx:
            gc_seeds_list_of_lists.append([s_index])
            gc_targets_list_of_lists.append([t_index])
            
    gc_indices = (gc_seeds_list_of_lists, gc_targets_list_of_lists)

    # --- FIX for 'rank' argument (Critical for GC stability) ---
    n_cons = len(gc_seeds_list_of_lists) 
    model_order = 1  
    rank_array = np.array([
        np.full(n_cons, model_order), 
        np.full(n_cons, model_order)  
    ])

    all_gc_results = []
    for subject, epochs in all_epochs_data.items():
        try:
            # Compute GC
            gc_con = spectral_connectivity_epochs(
                epochs, method='gc', indices=gc_indices,
                fmin=FREQUENCY_BANDS['mu'][0],
                fmax=FREQUENCY_BANDS['beta'][1], # Combined Mu+Beta
                faverage=True, sfreq=sfreq, n_jobs=-1,
                rank=rank_array  
            )
            
            # Reshape 1D result back to 3x14
            gc_data_1d = gc_con.get_data().squeeze()
            gc_matrix_3x14 = gc_data_1d.reshape((len(seed_idx), len(target_idx)))
            
            all_gc_results.append(gc_matrix_3x14)
            
        except Exception as e:
            print(f"! Subject {subject} GC failed: {e}")

    # --- Start of Analysis & Plotting ---
    if all_gc_results:
        avg_gc = np.mean(all_gc_results, axis=0)

        # Create and save DataFrame
        gc_df = pd.DataFrame(avg_gc, 
                             index=[ch_names[i] for i in seed_idx], 
                             columns=[ch_names[i] for i in target_idx])
        csv_path_gc = os.path.join(DERIVATIVES_ROOT, 'group_granger_causality.csv')
        gc_df.to_csv(csv_path_gc)
        print(f"  > Saved Granger Causality to: {csv_path_gc}")
        print(f"  > Group-Average GC (8-30 Hz) from {seed_channels} to Targets:")
        print(gc_df.to_string(float_format="%.4f"))

        # --- PLOT: GRANGER CAUSALITY (Directed Plot) ---
        GC_THRESHOLD = 0.007 
        gc_matrix = gc_df.values
        all_channel_names = ch_names 
        n_channels = len(all_channel_names)

        # 1. FIX: RE-IMPLEMENT CIRCULAR LAYOUT (Guarantees (N, 2) output)
        angles = np.linspace(0, 2 * np.pi, n_channels, endpoint=False)
        node_pos_array = np.vstack([np.cos(angles), np.sin(angles)]).T
        
        # 2. Create the position dictionary
        node_pos = {name: pos for name, pos in zip(all_channel_names, node_pos_array)}
        
        # 3. Create the NetworkX Graph and add Edges
        G_gc = nx.DiGraph()
        G_gc.add_nodes_from(all_channel_names)
        
        for i, seed in enumerate(gc_df.index):
            for j, target in enumerate(gc_df.columns):
                weight = gc_matrix[i, j]
                if weight > GC_THRESHOLD:
                    G_gc.add_edge(seed, target, weight=weight)
                    
        # 4. Prepare drawing parameters
        graph_node_list = list(G_gc.nodes())
        edge_weights = [d['weight'] for (u, v, d) in G_gc.edges(data=True)]
        edge_widths = [w * 250 for w in edge_weights] 
        
        # 5. Color map
        DEFAULT_COLOR = '#E0E0E0' 
        node_color_map = {
            node: '#87CEFA' if node in seed_channels else DEFAULT_COLOR 
            for node in all_channel_names
        }
        color_list = [node_color_map[node] for node in graph_node_list]
        
        # 6. Extract X/Y Coordinates 
        x_coords = node_pos_array[:, 0]
        y_coords = node_pos_array[:, 1]
        
        # --- Drawing ---
        fig_gc, ax_gc = plt.subplots(figsize=(8, 8))
        
        # A. DRAW NODES (Scatter Plot - Bypassing networkx bug)
        ax_gc.scatter(x_coords, y_coords,
                      s=1200,          
                      c=color_list,    
                      edgecolors='black', 
                      zorder=3)        
                            
        # B. DRAW EDGES (NetworkX)
        nx.draw_networkx_edges(G_gc, node_pos, 
                              edgelist=G_gc.edges(),
                              width=edge_widths,
                              arrowsize=15, 
                              
                              # CRITICAL FIX: Clip the alpha value to max 1.0
                              edge_color=[(1, 0, 0, min(w * 50, 1.0)) for w in edge_weights],
                              
                              ax=ax_gc,
                              connectionstyle='arc3,rad=0.15')

        # C. DRAW LABELS (NetworkX)
        labels = {node: node for node in graph_node_list}
        nx.draw_networkx_labels(G_gc, node_pos, labels=labels, font_size=10, ax=ax_gc, font_color='black')
        
        # D. Final Formatting and Saving
        ax_gc.set_title(f"Group-Average Granger Causality Flow (Threshold > {GC_THRESHOLD:.3f})")
        ax_gc.set_xlim(min(x_coords) - 0.2, max(x_coords) + 0.2)
        ax_gc.set_ylim(min(y_coords) - 0.2, max(y_coords) + 0.2)
        ax_gc.axis('off')
        
        plt.tight_layout()
        fig_gc.savefig(os.path.join(FIGURES_ROOT, 'group_granger_causality_flow.png'), dpi=300)
        plt.show(block=False)
         
    else:
        print("  > Granger Causality analysis could not be completed.")       
        
    return 



def run_nbs_analysis(all_epochs_data, sfreq, summary_results, connectivity_method, fmin, fmax, analysis_name):
    print(f"\n[Analysis 2.4] Computing NBS ({analysis_name})...")

    # 1. Create connectivity matrices
    con_mi_matrices = []
    con_idle_matrices = []
    successful_subjects = []

    for subject, epochs in all_epochs_data.items():
        try:
            # --- SCIENTIFIC CORRECTION: Group events into MI vs. Idle ---
            # 1. Create MI epochs (Events 1 and 3)
            epochs_mi = mne.epochs.concatenate_epochs([epochs['1'], epochs['3']])
            if len(epochs_mi) < 2: 
                print(f"! Subject {subject} NBS prep failed: Fewer than 2 MI epochs.")
                continue
                
            # 2. Create Idle epochs (Events 2 and 4)
            epochs_idle = mne.epochs.concatenate_epochs([epochs['2'], epochs['4']])
            if len(epochs_idle) < 2:
                print(f"! Subject {subject} NBS prep failed: Fewer than 2 Idle epochs.")
                continue
            # -----------------------------------------------------------

            # Check for NaN/Inf
            if not np.all(np.isfinite(epochs_mi.get_data())) or not np.all(np.isfinite(epochs_idle.get_data())):
                print(f"! Subject {subject} NBS prep failed: Data contains NaN or Inf values.")
                continue
            
            # Compute connectivity for MI
            con_mi = spectral_connectivity_epochs(
                epochs_mi, method=connectivity_method,
                fmin=fmin, fmax=fmax,
                faverage=True, sfreq=sfreq, n_jobs=-1
            )
            # Compute connectivity for Idle
            con_idle = spectral_connectivity_epochs(
                epochs_idle, method=connectivity_method,
                fmin=fmin, fmax=fmax,
                faverage=True, sfreq=sfreq, n_jobs=-1
            )
            
            con_mi_matrices.append(con_mi.get_data(output='dense')[:, :, 0])
            con_idle_matrices.append(con_idle.get_data(output='dense')[:, :, 0])
            successful_subjects.append(subject)

        except Exception as e:
            print(f"! Subject {subject} NBS prep failed with error: {e}")

    # 2. Stack matrices
    if len(con_mi_matrices) >= 2 and len(con_idle_matrices) == len(con_mi_matrices):
        X_mi = np.stack(con_mi_matrices)
        X_idle = np.stack(con_idle_matrices)
        
        # 3. Perform NBS
        t_thresh = 2.04  # Critical t-value
        n_perms = 5000
        n_subjects_used = X_mi.shape

        print(f"  > Running NBS paired-sample t-test (N={n_subjects_used}) with {n_perms} permutations...")
        
        try:
            # Add tiny noise to break zero variance
            rng = np.random.default_rng(seed=ICA_RANDOM_STATE)
            noise_mi = rng.normal(loc=0, scale=1e-6, size=X_mi.shape)
            noise_idle = rng.normal(loc=0, scale=1e-6, size=X_idle.shape)
            
            X_mi_T = np.transpose(X_mi + noise_mi, (1, 2, 0))
            X_idle_T = np.transpose(X_idle + noise_idle, (1, 2, 0))

            # Call bct.nbs_bct with paired=True
            pval, adj_matrix, null_dist = bct.nbs_bct(
                X_mi_T, X_idle_T,  # Compare MI vs Idle
                thresh=t_thresh,
                k=n_perms,
                tail='both',
                paired=True
            )
            
            n_sig_components = len(pval)
            print("  > NBS Results:")
            if n_sig_components > 0:
                print(f"   - Found {n_sig_components} component(s) (pre-significance).")
                
                sorted_indices = np.argsort(pval)
                p_min = pval[sorted_indices[0]]
                sig_adj_matrix = adj_matrix[sorted_indices[0]]
                n_links = np.sum(sig_adj_matrix > 0) / 2
                
                print(f"   - Most significant component: p-value = {p_min:.4f}, Links = {int(n_links)}")
                
                summary_results[f'nbs_p_value_{analysis_name}'] = [p_min]
                summary_results[f'nbs_n_links_{analysis_name}'] = [int(n_links)]
                
                if p_min < 0.05 and n_links > 0:
                    print("   - Plotting significant network...")
                    sig_adj_matrix = adj_min
                    node_names = RESULTS_CACHE['ch_names']
                    
                    n_channels = len(node_names)
                    angles = np.linspace(0, 2 * np.pi, n_channels, endpoint=False)
                    node_angles_array = np.vstack([np.cos(angles), np.sin(angles)]).T

                    fig = plot_connectivity_circle(
                        sig_adj_matrix,
                        node_names,
                        n_lines=None,
                        node_angles=node_angles_array,
                        node_colors='skyblue',
                        node_edgecolor='black',
                        title=f'NBS: Significant Network ({analysis_name}, p={p_min:.4f})',
                        show=False
                    )
                    fig.set_size_inches(6, 6)
                    fig.savefig(os.path.join(FIGURES_ROOT, f'group_nbs_network_{analysis_name}.png'), dpi=300)
                    plt.show(block=False)
                else:
                    print(f"   - No significant network found (p={p_min:.4f}). Skipping plot.")
                
            else:
                print(f"   - No components found at t-threshold={t_thresh}")
                summary_results[f'nbs_p_value_{analysis_name}'] = [1.0] # Store non-sig p-value
                summary_results[f'nbs_n_links_{analysis_name}'] = [0]

        except Exception as e:
            print(f"! FAILED: NBS analysis error. {e}")
            
    else:
        print("  > Not enough valid data to perform NBS analysis (e.g., mismatched subject counts or < 2 subjects).")

    return summary_results



def run_decoding(all_epochs_data, summary_results, ICA_RANDOM_STATE, DERIVATIVES_ROOT, FIGURES_ROOT, RESULTS_CACHE):
    """
    MODIFIED: Runs CSP + multiple classifiers (LDA, SVM, RF) with a stability loop.
    
    Compares classifier performance and saves the individual scores for the
    primary classifier (LDA) for use in correlation analyses.
    """

    print(" \n[Analysis 2.5] Performing Subject-Specific Decoding (CSP + Classifier Benchmarking)...")
    
    # --- NEW: Define classifiers to test ---
    classifiers_to_test = {
        'LDA': LinearDiscriminantAnalysis(),
        'SVM': SVC(kernel='rbf'), # 'rbf' is a good non-linear default
        'RF': RandomForestClassifier(n_estimators=100, random_state=ICA_RANDOM_STATE)
    }

    # --- NEW: Create a dict to store all results ---
    all_classifier_accuracies = {name: [] for name in classifiers_to_test}
    all_subject_indices = [] # To store subject IDs

    # --- NEW: Define the stability parameter ---
    N_ITERATIONS = 30 # From your original script
    # -----------------------------------------

    for subject, epochs_decoding in all_epochs_data.items():
        print(f"  > Decoding for Subject: {subject}")
        
        try:
            # --- (Your existing data prep - NO CHANGES) ---
            epochs_mi = mne.epochs.concatenate_epochs([epochs_decoding['1'], epochs_decoding['3']])
            epochs_mi.events[:, -1] = 10  
            epochs_idle = mne.epochs.concatenate_epochs([epochs_decoding['2'], epochs_decoding['4']])
            epochs_idle.events[:, -1] = 20
            n = min(len(epochs_mi), len(epochs_idle))
            epochs_mi = epochs_mi[:n]
            epochs_idle = epochs_idle[:n]
            X_mi_data = epochs_mi.get_data(picks='eeg')
            X_idle_data = epochs_idle.get_data(picks='eeg')
            X = np.concatenate([X_mi_data, X_idle_data])
            y = np.array([1] * n + [2] * n)
            if not np.all(np.isfinite(X)):
                print(f" ! FAILED decoding for subject {subject}. Data contains NaN or Inf.")
                continue
            if len(np.unique(y)) < 2:
                print(f" ! FAILED decoding for subject {subject}. Missing one class of data.")
                continue
            # --- (End of existing data prep) ---

            # --- NEW: Loop through each classifier for this subject ---
            for clf_name, clf_model in classifiers_to_test.items():
                
                csp = CSP(n_components=4, reg='ledoit_wolf', log=True, norm_trace=False)
                
                # Create the pipeline for the current classifier
                clf_pipeline = Pipeline([
                    ('csp', csp),
                    (clf_name, clf_model) # Use the classifier from the loop
                ])

                # --- (Your existing stability loop - NO CHANGES) ---
                iteration_accuracies = []
                # print(f"    > {clf_name}: Running stability analysis (N={N_ITERATIONS} runs)...") # Optional print
                
                for i in range(N_ITERATIONS):
                    current_seed = i 
                    cv_stable = ShuffleSplit(n_splits=10, test_size=0.2, random_state=current_seed)
                    scores_iter = cross_val_score(clf_pipeline, X, y, cv=cv_stable, n_jobs=-1)
                    iteration_accuracies.append(np.mean(scores_iter))

                subject_mean_accuracy = np.mean(iteration_accuracies)
                # --- (End of existing stability loop) ---
                
                # Store this subject's stable mean for this classifier
                all_classifier_accuracies[clf_name].append(subject_mean_accuracy)

            # --- END of new classifier loop ---
            
            all_subject_indices.append(int(subject)) # Add subject index once

        except Exception as e:
            print(f" ! FAILED decoding for subject {subject}. Error: {e}")
            continue 
            
    if not all_subject_indices:
        print("  > Decoding analysis could not be completed (no subjects processed).")
        return summary_results

    # --- NEW: Group-level summary and plotting ---
    print("\n--- Classifier Benchmarking Summary ---")
    summary_data = []
    
    for clf_name, accuracies in all_classifier_accuracies.items():
        if accuracies:
            group_mean = np.mean(accuracies)
            t_stat, p_val = ttest_1samp(accuracies, popmean=0.5, alternative='greater')
            print(f"  > {clf_name}: Group Mean = {group_mean*100:.2f}% (p={p_val:.4f})")
            
            summary_data.append({
                'classifier': clf_name,
                'mean_accuracy_pct': group_mean * 100,
                't_statistic': t_stat,
                'p_value': p_val
            })
            
            # --- SAVE INDIVIDUAL SCORES (CRITICAL) ---
            # We save the scores for each classifier to a *unique file*
            decoding_df = pd.DataFrame({
                'subject_index': all_subject_indices,
                'accuracy': accuracies 
            })
            csv_path_decoding = os.path.join(DERIVATIVES_ROOT, f'individual_decoding_scores_{clf_name}.csv')
            decoding_df.to_csv(csv_path_decoding, index=False)
            print(f"    > Saved individual {clf_name} scores to: {csv_path_decoding}")
            
            # Store the LDA scores in the cache for the *original* correlation
            if clf_name == 'LDA':
                RESULTS_CACHE['decoding_df'] = decoding_df 

    print("-----------------------------------------")
    
    # Save summary table
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(DERIVATIVES_ROOT, 'classifier_comparison_summary.csv'), index=False)

    # --- NEW PLOT: Classifier Comparison Bar Chart ---
    try:
        fig_comp, ax_comp = plt.subplots(figsize=(8, 5))
        ax_comp.bar(summary_df['classifier'], summary_df['mean_accuracy_pct'], 
                    color=['#007acc', '#4caf50', '#f44336'])
        ax_comp.axhline(50, color='black', linestyle='dotted', label='Chance (50%)')
        ax_comp.set_ylabel('Group Mean Accuracy (%)')
        ax_comp.set_ylim(0, 100)
        ax_comp.set_title('BCI Classifier Performance Comparison')
        for i, row in summary_df.iterrows():
            ax_comp.text(i, row['mean_accuracy_pct'] + 1, f"{row['mean_accuracy_pct']:.2f}%", 
                         ha='center', fontweight='bold')
        ax_comp.legend()
        plt.tight_layout()
        fig_comp.savefig(os.path.join(FIGURES_ROOT, 'report_classifier_comparison.png'), dpi=300)
        plt.show(block=False)
    except Exception as e:
        print(f"! FAILED to plot classifier comparison: {e}")

    # --- (Your original plots, now based on LDA results) ---
    try:
        # Get LDA results for plotting
        lda_accuracies = all_classifier_accuracies.get('LDA', [])
        if lda_accuracies:
            group_mean_lda = np.mean(lda_accuracies)
            
            # --- PLOT: Decoding Accuracy Bar Chart (LDA) ---
            fig_decoding, ax_decoding = plt.subplots(figsize=(5, 5))
            accuracies = [group_mean_lda * 100, 50.0]
            labels = ['LDA Mean Accuracy', 'Chance Level']
            ax_decoding.bar(labels, accuracies, color=['royalblue', 'gray'])
            ax_decoding.set_ylabel('Accuracy (%)')
            ax_decoding.set_ylim(0, 100)
            ax_decoding.set_title('Group-Average Decoding Accuracy (Stable Mean - LDA)')
            for i, acc in enumerate(accuracies):
                ax_decoding.text(i, acc + 2, f'{acc:.2f}%', ha='center', fontweight='bold')
            plt.tight_layout()
            fig_decoding.savefig(os.path.join(FIGURES_ROOT, 'report_decoding_accuracy_LDA.png'), dpi=300)
            plt.show(block=False)

            # --- PLOT: Histogram of Stable Subject Accuracies (LDA) ---
            fig_hist, ax_hist = plt.subplots(figsize=(8, 5))
            ax_hist.hist(lda_accuracies, bins=8, edgecolor='black', alpha=0.7)
            ax_hist.axvline(group_mean_lda, color='red', linestyle='dashed', linewidth=2,
                            label=f'Group Stable Mean: {group_mean_lda*100:.2f}%')
            ax_hist.axvline(0.5, color='black', linestyle='dotted', linewidth=2,
                            label='Chance Level: 50.00%')
            ax_hist.set_title('Distribution of Stable BCI Accuracies (LDA, N=32 Subjects)')
            ax_hist.set_xlabel('Stable Mean Accuracy (from 30 CV runs per subject)')
            ax_hist.set_ylabel('Number of Subjects')
            ax_hist.legend()
            ax_hist.grid(axis='y', linestyle='--', alpha=0.7)
            ax_hist.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
            plt.tight_layout()
            fig_hist.savefig(os.path.join(FIGURES_ROOT, 'report_decoding_accuracy_distribution_LDA.png'), dpi=300)
            plt.show(block=False)
            
            # --- PLOT: CSP Spatial Patterns (same as original) ---
            # (This part of your code is fine as is, it just needs the 'all_epochs_data')
            subjects_to_plot = list(all_epochs_data.keys())[:3]
            for plot_subject_id in subjects_to_plot:
                try:
                    epochs_plot_mi = mne.epochs.concatenate_epochs([all_epochs_data[plot_subject_id]['1'], all_epochs_data[plot_subject_id]['3']])
                    epochs_plot_idle = mne.epochs.concatenate_epochs([all_epochs_data[plot_subject_id]['2'], all_epochs_data[plot_subject_id]['4']])
                    n_plot = min(len(epochs_plot_mi), len(epochs_plot_idle))
                    epochs_plot_mi = epochs_plot_mi[:n_plot]
                    epochs_plot_idle = epochs_plot_idle[:n_plot]
                    X_plot = np.concatenate([
                        epochs_plot_mi.get_data(picks='eeg'), 
                        epochs_plot_idle.get_data(picks='eeg')
                    ])
                    y_plot = np.array([1] * n_plot + [2] * n_plot)
                    
                    csp_plot = CSP(n_components=4, reg='ledoit_wolf', log=True, norm_trace=False)
                    csp_plot.fit(X_plot, y_plot)
                    
                    fig_csp = csp_plot.plot_patterns(epochs_plot_mi.info, ch_type='eeg', 
                                                     units='Patterns (AU)', size=1.5, show=False, 
                                                     colorbar=False)
                    fig_csp.suptitle(f"CSP Patterns: Subject {plot_subject_id}")
                    plt.tight_layout(rect=[0, 0, 1, 0.95])
                    filename = f'report_csp_patterns_sub-{plot_subject_id}.png'
                    fig_csp.savefig(os.path.join(FIGURES_ROOT, filename), dpi=300)
                    plt.show(block=False)
                except Exception as e:
                    print(f" ! FAILED to plot CSP patterns for {plot_subject_id}: {e}")
            # --- END OF PLOT ---
    
    except Exception as e:
        print(f"! FAILED to generate LDA plots: {e}")
        
    return summary_results # Pass back the original summary_results (or update if needed)



def run_correlation_analysis():
    """
    Loads individual LDA decoding scores and SENSOR graph metrics, 
    then runs a Pearson correlation analysis between them.
    
    MODIFIED: Now specifically loads 'individual_decoding_scores_LDA.csv'
    and adds a scatter plot.
    """
    print("\n" + "="*50)
    print("[Analysis 2.6] Running SENSOR-SPACE Correlation Analysis")
    print("="*50)

    # 1. Load the individual decoding scores
    # --- MODIFIED LINE ---
    decoding_scores_file = os.path.join(DERIVATIVES_ROOT, 'individual_decoding_scores_LDA.csv')
    if not os.path.exists(decoding_scores_file):
        print("! ERROR: 'individual_decoding_scores_LDA.csv' not found.")
        print("! Please run Decoding (option 7) first.")
        return
    # --- END MODIFICATION ---
    
    decoding_df = pd.read_csv(decoding_scores_file)

    # 2. Load the individual global metrics (Sensor)
    gmetrics_file = os.path.join(DERIVATIVES_ROOT, 'individual_global_metrics.csv')
    if not os.path.exists(gmetrics_file):
        print("! ERROR: 'individual_global_metrics.csv' not found.")
        print("! Please run Connectivity (option 3) first.")
        return
    global_metrics_df = pd.read_csv(gmetrics_file)

    # 3. Load and aggregate all individual NODAL metrics (Sensor)
    all_fc1_betweenness = []
    all_p3_betweenness = []
    all_f3_strength = []
    
    subjects_in_order = decoding_df['subject_index'].astype(str).str.zfill(3)
    
    for subject_id in subjects_in_order:
        
        subject_dir = os.path.join(DERIVATIVES_ROOT, f'sub-{subject_id}')
        nodal_file = os.path.join(
            subject_dir, 
            f'sub-{subject_id}_desc-Beta_PLV_nodal_metrics.csv' # Sensor file
        )
        
        if not os.path.exists(nodal_file):
            print(f"  > Warning: Missing nodal file for sub-{subject_id}. Skipping.")
            all_fc1_betweenness.append(np.nan)
            all_p3_betweenness.append(np.nan)
            all_f3_strength.append(np.nan)
            continue
            
        nodal_df = pd.read_csv(nodal_file)
        
        try:
            fc1_b = nodal_df.loc[nodal_df['Channel'] == 'FC1', 'Betweenness_Centrality'].values
            p3_b = nodal_df.loc[nodal_df['Channel'] == 'P3', 'Betweenness_Centrality'].values
            f3_s = nodal_df.loc[nodal_df['Channel'] == 'F3', 'Nodal_Strength'].values
            
            all_fc1_betweenness.append(fc1_b[0] if fc1_b.size > 0 else np.nan)
            all_p3_betweenness.append(p3_b[0] if p3_b.size > 0 else np.nan)
            all_f3_strength.append(f3_s[0] if f3_s.size > 0 else np.nan)
        except Exception as e:
            print(f"  > Error processing nodal file for {subject_id}: {e}")
            all_fc1_betweenness.append(np.nan)
            all_p3_betweenness.append(np.nan)
            all_f3_strength.append(np.nan)

    # 4. Combine all data into one DataFrame
    final_df = pd.merge(decoding_df, global_metrics_df, on='subject_index', how='left')
    final_df['FC1_Betweenness'] = all_fc1_betweenness
    final_df['P3_Betweenness'] = all_p3_betweenness
    final_df['F3_Nodal_Strength'] = all_f3_strength
    
    final_df_clean = final_df.dropna()
    print(f"  > Correlating metrics for {len(final_df_clean)} subjects with complete data.")

    if len(final_df_clean) < 2:
        print("! ERROR: Not enough data to run correlation. (Need at least 2 subjects).")
        return

    print("\n--- SENSOR Correlation Analysis Results (Pearson's r) ---")
    
    # 5. Run and print the statistical correlations
    
    # Correlation: Accuracy vs. Global Efficiency
    corr_glob_eff, p_val_glob_eff = pearsonr(final_df_clean['accuracy'], final_df_clean['Beta_PLV_Global_Efficiency'])
    print(f"Accuracy vs. Global Efficiency (Beta PLV): r = {corr_glob_eff:.3f}, p = {p_val_glob_eff:.4f}")

    # Correlation: Accuracy vs. 'Bridge' Node (FC1)
    corr_fc1, p_val_fc1 = pearsonr(final_df_clean['accuracy'], final_df_clean['FC1_Betweenness'])
    print(f"Accuracy vs. FC1 Betweenness (Beta PLV): r = {corr_fc1:.3f}, p = {p_val_fc1:.4f}")

    # Correlation: Accuracy vs. 'Bridge' Node (P3)
    corr_p3, p_val_p3 = pearsonr(final_df_clean['accuracy'], final_df_clean['P3_Betweenness'])
    print(f"Accuracy vs. P3 Betweenness (Beta PLV): r = {corr_p3:.3f}, p = {p_val_p3:.4f}")
    
    # Correlation: Accuracy vs. 'Hub' Node (F3)
    corr_f3, p_val_f3 = pearsonr(final_df_clean['accuracy'], final_df_clean['F3_Nodal_Strength'])
    print(f"Accuracy vs. F3 Nodal Strength (Beta PLV): r = {corr_f3:.3f}, p = {p_val_f3:.4f}")
    
    print("--------------------------------------------------\n")
    
    final_stats_file = os.path.join(DERIVATIVES_ROOT, 'individual_all_metrics_and_scores.csv')
    final_df_clean.to_csv(final_stats_file, index=False)
    print(f"  > Final correlation data saved to {final_stats_file}")

    # --- ADDED: PLOT SENSOR-SPACE CORRELATION ---
    try:
        fig_corr_sensor, ax_corr_sensor = plt.subplots(figsize=(7, 5))
        ax_corr_sensor.scatter(final_df_clean['Beta_PLV_Global_Efficiency'], final_df_clean['accuracy'], 
                        alpha=0.7, edgecolors='k')
        
        # Add trend line
        z = np.polyfit(final_df_clean['Beta_PLV_Global_Efficiency'], final_df_clean['accuracy'], 1)
        p = np.poly1d(z)
        ax_corr_sensor.plot(final_df_clean['Beta_PLV_Global_Efficiency'], p(final_df_clean['Beta_PLV_Global_Efficiency']), "r--")
        
        ax_corr_sensor.set_title(f"Sensor-Space Correlation (r={corr_glob_eff:.3f}, p={p_val_glob_eff:.4f})")
        ax_corr_sensor.set_xlabel("Beta-PLV Global Efficiency (Sensor Space)")
        ax_corr_sensor.set_ylabel("BCI Decoding Accuracy (LDA)")
        ax_corr_sensor.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        plt.tight_layout()
        fig_corr_sensor.savefig(os.path.join(FIGURES_ROOT, 'report_correlation_accuracy_vs_sensor_glob_eff.png'), dpi=300)
        plt.show(block=False)
    except Exception as e:
        print(f"! FAILED to plot sensor correlation: {e}")
        
    return
    

def run_MI_vs_Idle_network_comparison():
    """
    Solves the "NBS Null-Result vs. Graph Theory" problem.
    
    This function calculates graph metrics for BOTH MI and Idle states
    for every subject, saves them to a CSV, runs a paired t-test,
    and generates a bar plot of the mean differences.
    """
    print("\n" + "="*50)
    print("[Analysis 2.7] Running MI vs. Idle Network Comparison")
    print("="*50)
    
    if not RESULTS_CACHE['all_epochs_data']:
        print("! ERROR: Epochs not loaded. Please run step 2 (Preprocessing) first.")
        return

    ch_names = RESULTS_CACHE['ch_names']
    sfreq = RESULTS_CACHE['sfreq']
    
    # --- ADD THIS BLOCK ---
    # Get the integer index for the channels of interest
    try:
        ch_idx_f3 = ch_names.index('F3')
        ch_idx_fc1 = ch_names.index('FC1')
        ch_idx_p3 = ch_names.index('P3')
    except ValueError as e:
        print(f"! ERROR: A required channel (F3, FC1, P3) is not in ch_names. {e}")
        return
    # --- END OF BLOCK ---]
    
    all_metrics = []
    
    for subject, epochs in RESULTS_CACHE['all_epochs_data'].items():
        print(f"  > Processing networks for Subject: {subject}")
        try:
            epochs_mi = mne.epochs.concatenate_epochs([epochs['1'], epochs['3']])
            epochs_idle = mne.epochs.concatenate_epochs([epochs['2'], epochs['4']])
        except KeyError:
            print(f" ! Skipping Subject {subject}: Missing one or more event types.")
            continue
        except ValueError:
            print(f" ! Skipping Subject {subject}: No data for one or more event types.")
            continue

        try:
            # --- Calculate MI Connectivity & Metrics ---
            con_mi = spectral_connectivity_epochs(epochs_mi, method='plv',
                fmin=FREQUENCY_BANDS['beta'][0], fmax=FREQUENCY_BANDS['beta'][1],
                faverage=True, sfreq=sfreq, n_jobs=-1)
            con_mi_data = (con_mi.get_data(output='dense').squeeze() + con_mi.get_data(output='dense').squeeze().T) / 2
            
            L_mi = 1. / (con_mi_data + 1e-6); np.fill_diagonal(L_mi, 0)
            
            # --- Calculate Idle Connectivity & Metrics ---
            con_idle = spectral_connectivity_epochs(epochs_idle, method='plv',
                fmin=FREQUENCY_BANDS['beta'][0], fmax=FREQUENCY_BANDS['beta'][1],
                faverage=True, sfreq=sfreq, n_jobs=-1)
            con_idle_data = (con_idle.get_data(output='dense').squeeze() + con_idle.get_data(output='dense').squeeze().T) / 2
            
            L_idle = 1. / (con_idle_data + 1e-6); np.fill_diagonal(L_idle, 0)

            # --- Store metrics for this subject ---
            all_metrics.append({
                'subject': subject,
                'MI_Global_Eff': bct.efficiency_wei(con_mi_data),
                'Idle_Global_Eff': bct.efficiency_wei(con_idle_data),
                
                # --- FIX: Get the single channel value using the index ---
                'MI_F3_Strength': bct.strengths_und(con_mi_data)[ch_idx_f3],
                'Idle_F3_Strength': bct.strengths_und(con_idle_data)[ch_idx_f3],
                'MI_FC1_Centrality': bct.betweenness_wei(L_mi)[ch_idx_fc1],
                'Idle_FC1_Centrality': bct.betweenness_wei(L_idle)[ch_idx_fc1],
                'MI_P3_Centrality': bct.betweenness_wei(L_mi)[ch_idx_p3],
                'Idle_P3_Centrality': bct.betweenness_wei(L_idle)[ch_idx_p3]
            })

        except Exception as e:
            print(f"! FAILED processing for subject {subject}. Error: {e}")

    if not all_metrics:
        print("! Analysis failed: No subjects were processed.")
        return

    # --- Collate data, Save to CSV, and Run Stats ---
    df = pd.DataFrame(all_metrics)
    
    # --- 1. SAVE CSV ---
    csv_path = os.path.join(DERIVATIVES_ROOT, 'individual_mi_vs_idle_network_metrics.csv')
    df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"\n  > Saved individual MI vs. Idle metrics to: {csv_path}")

    print("\n--- Paired T-Test Results (MI vs. Idle, Beta PLV) ---")
    
    def run_paired_ttest(a, b, name):
        try:
            t_stat, p_val = ttest_1samp(a - b, 0)
            print(f"{name}: t = {t_stat:.3f}, p = {p_val:.4f} {'***' if p_val < 0.05 else ''}")
            return np.mean(a), np.mean(b)
        except Exception as e:
            print(f"! T-test failed for {name}: {e}")
            return np.nan, np.nan

    mean_metrics = {}
    mean_metrics['Global_Eff'] = run_paired_ttest(df['MI_Global_Eff'], df['Idle_Global_Eff'], "Global Efficiency")
    mean_metrics['F3_Strength'] = run_paired_ttest(df['MI_F3_Strength'], df['Idle_F3_Strength'], "F3 Nodal Strength")
    mean_metrics['FC1_Centrality'] = run_paired_ttest(df['MI_FC1_Centrality'], df['Idle_FC1_Centrality'], "FC1 Betweenness Centrality")
    mean_metrics['P3_Centrality'] = run_paired_ttest(df['MI_P3_Centrality'], df['Idle_P3_Centrality'], "P3 Betweenness Centrality")
    print("-------------------------------------------------------")

    # --- 2. CREATE PLOT ---
    labels = ['Global_Eff', 'F3_Strength', 'FC1_Centrality', 'P3_Centrality']
    mi_means = [mean_metrics[m][0] for m in labels]
    idle_means = [mean_metrics[m][1] for m in labels]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, mi_means, width, label='Motor Imagery', color='royalblue')
    rects2 = ax.bar(x + width/2, idle_means, width, label='Idle', color='gray')

    ax.set_ylabel('Mean Metric Value')
    ax.set_title('Network Metric Comparison: Motor Imagery vs. Idle (Beta PLV)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    ax.bar_label(rects1, padding=3, fmt='%.3f')
    ax.bar_label(rects2, padding=3, fmt='%.3f')
    
    fig.tight_layout()
    
    plot_path = os.path.join(FIGURES_ROOT, 'group_mi_vs_idle_comparison.png')
    fig.savefig(plot_path, dpi=300)
    print(f"  > Saved comparison plot to: {plot_path}")
    plt.close(fig) # Close figure
    
    
def run_source_modeling():
    """
    [NEW ANALYSIS - CORRECTED v5] Computes source-space time courses for all subjects using eLORETA.
    This function reads from RESULTS_CACHE['all_epochs_data'] and saves
    its output to RESULTS_CACHE['all_label_epochs'].
    This is a time-consuming step.
    
    (Version 5: Fixes MemoryError by adding gc.collect() and limiting n_jobs)
    """
    print("\n" + "="*50)
    print("[Analysis 3.1] Running Source Modeling (eLORETA)")
    print("="*50)

    if 'all_epochs_data' not in RESULTS_CACHE or not RESULTS_CACHE['all_epochs_data']:
        print("! ERROR: Epochs not loaded. Please run step 2 (Preprocessing) first.")
        return False
        
    all_epochs_data = RESULTS_CACHE['all_epochs_data']

    # --- 1. Setup FS Average Source Space ---
    try:
        print("  > Fetching/loading fsaverage template...")
        fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
        subjects_dir = os.path.dirname(fs_dir)
        src = mne.setup_source_space('fsaverage', spacing='oct6', subjects_dir=subjects_dir, add_dist=False, verbose=False)
        
        # --- 2. *CREATE* Template BEM ---
        print("  > Creating 3-layer BEM model (this may take a moment)...")
        bem_surfaces = mne.make_bem_model(subject='fsaverage', ico=4,
                                          subjects_dir=subjects_dir,
                                          conductivity=(0.3, 0.006, 0.3), # Standard for EEG
                                          verbose=False)
        bem = mne.make_bem_solution(bem_surfaces, verbose=False)

        # --- 3. Load Parcellation (ROIs) ---
        labels = mne.read_labels_from_annot('fsaverage', parc='aparc', subjects_dir=subjects_dir)
        labels = [label for label in labels if 'unknown' not in label.name]
        label_names = [label.name for label in labels]
        print(f"  > Loaded 'aparc' atlas with {len(label_names)} ROIs.")

    except Exception as e:
        print(f"! CRITICAL ERROR: Could not load or create fsaverage template/atlas. {e}")
        import traceback
        traceback.print_exc()
        return False

    # --- 4. Process Each Subject ---
    all_label_epochs = {}
    for subject, epochs in all_epochs_data.items():
        print(f"  > Modeling source space for Subject: {subject}")
        try:
            info = epochs.info.copy()
            montage = mne.channels.make_standard_montage('standard_1020')
            info.set_montage(montage, on_missing='ignore')
            
            # --- 5. Compute Forward Model ---
            # --- MEMORY FIX 1: Set n_jobs to 4 (safer than -1) ---
            fwd = mne.make_forward_solution(info, trans='fsaverage', src=src, bem=bem, 
                                            eeg=True, mindist=5.0, n_jobs=4, verbose=False)

            # --- 6. Compute Inverse Model (eLORETA) ---
            cov = mne.compute_covariance(epochs, tmin=EPOCH_TMIN, tmax=EPOCH_TMAX, 
                                         method='shrunk', rank=None, verbose=False)
            inverse_operator = make_inverse_operator(info, fwd, cov, loose=0.2, depth=0.8, verbose=False)

            # --- 7. Apply Inverse Model & Extract ROIs ---
            snr = 3.0
            lambda2 = 1.0 / snr ** 2
            
            stcs = apply_inverse_epochs(epochs, inverse_operator, lambda2,
                                        method="eLORETA", pick_ori="normal",
                                        return_generator=False, verbose=False)
            
            label_ts = mne.extract_label_time_course(stcs, labels, src, mode='mean_flip',
                                                     return_generator=False, verbose=False)
            
            # --- 8. Create a new "Source Epochs" object ---
            info_src = mne.create_info(ch_names=label_names, sfreq=info['sfreq'], ch_types='eeg')
            src_epochs_data = np.array(label_ts) 
            source_epochs = mne.EpochsArray(src_epochs_data, info_src, events=epochs.events, 
                                            tmin=epochs.tmin, event_id=epochs.event_id,
                                            baseline=epochs.baseline)
            
            all_label_epochs[subject] = source_epochs
            
            # --- MEMORY FIX 2: Explicitly delete large variables ---
            del info, fwd, cov, inverse_operator, stcs, label_ts, src_epochs_data, source_epochs
            
        except Exception as e:
            print(f"! FAILED to model source space for {subject}: {e}")

        # --- MEMORY FIX 3: Run the garbage collector ---
        # This clears the memory from the loop before starting the next subject
        gc.collect()

    print("--- Source Modeling Complete ---")
    
    # Save a list of the ROI names for later
    RESULTS_CACHE['source_label_names'] = label_names
    RESULTS_CACHE['all_label_epochs'] = all_label_epochs
    
    return True


def run_source_connectivity():
    """
    [NEW ANALYSIS] This is a copy of 'run_connectivity' but adapted for
    source-space data from RESULTS_CACHE['all_label_epochs'].
    It saves all outputs with a '_SOURCE' suffix.
    """
    print("\n" + "="*50)
    print("[Analysis 3.2] Running Source-Space Connectivity")
    print("="*50)

    if 'all_label_epochs' not in RESULTS_CACHE or not RESULTS_CACHE['all_label_epochs']:
        print("! ERROR: Source epochs not found. Please run Source Modeling (step 14) first.")
        return
        
    all_label_epochs = RESULTS_CACHE['all_label_epochs']
    label_names = RESULTS_CACHE['source_label_names']
    sfreq = RESULTS_CACHE['sfreq'] # sfreq is the same
    
    connectivity_methods = ['plv', 'coh']
    all_connectivity_results = {'mu':[], 'beta':[]}
    all_subject_indices = []
    all_global_efficiencies = []
    
    for subject, epochs in all_label_epochs.items():
        print(f"  > Processing source connectivity for Subject: {subject}")
        
        try:
            epochs_mi = mne.epochs.concatenate_epochs([epochs['1'], epochs['3']])
        except (KeyError, ValueError):
            print(f" ! Skipping Subject {subject}: Missing or no data for event '1' or '3'.")
            continue

        # Mu Band (8-13 Hz) 
        con_mu = spectral_connectivity_epochs(
            epochs_mi, method=connectivity_methods,
            fmin=FREQUENCY_BANDS['mu'][0], fmax=FREQUENCY_BANDS['mu'][1],
            faverage=True, sfreq=sfreq, n_jobs=-1, verbose=False
        )
        # Beta Band (13-30 Hz) 
        con_beta = spectral_connectivity_epochs(
            epochs_mi, method=connectivity_methods,
            fmin=FREQUENCY_BANDS['beta'][0], fmax=FREQUENCY_BANDS['beta'][1],
            faverage=True, sfreq=sfreq, n_jobs=-1, verbose=False
        )
        all_connectivity_results['mu'].append(con_mu)
        all_connectivity_results['beta'].append(con_beta)

        # --- Calculate and Save Individual Graph Metrics (Source) ---
        try:
            con_beta_plv_data = con_beta[0].get_data(output='dense').squeeze()
            con_beta_plv_data = (con_beta_plv_data + con_beta_plv_data.T) / 2
            
            # NOTE: We re-use 'calculate_and_save_individual_graph_metrics'
            # but point it to new filenames by changing 'metric_name'
            global_eff = calculate_and_save_individual_graph_metrics(
                con_matrix=con_beta_plv_data,
                ch_names=label_names, # Use ROI names
                subject_id=subject,
                metric_name='Beta_PLV_SOURCE', # NEW SUFFIX
                output_dir=DERIVATIVES_ROOT 
            )
            
            if not np.isnan(global_eff):
                all_subject_indices.append(int(subject))
                all_global_efficiencies.append(global_eff)
            
        except Exception as e:
            print(f"! FAILED to calculate individual SOURCE graph metrics for {subject}: {e}")

    # --- Group Averaging (Source) ---
    try:
        if not all_connectivity_results['beta']:
            print("! FAILED: No source connectivity results were computed.")
            return

        # Beta PLV (Source)
        avg_beta_plv = np.mean([res[0].get_data(output='dense')[:, :, 0] for res in all_connectivity_results['beta']], axis=0)
        np.save(os.path.join(DERIVATIVES_ROOT, 'group_connect_beta_plv_SOURCE.npy'), avg_beta_plv)
        
        # (Add other bands like coh, mu if needed, following the same pattern)
        
        print("  > Source PLV group average computation complete.")
        
        # --- Plotting (Source) ---
        # Note: Plotting a 68x68 matrix with imshow is not very readable.
        # We will skip the imshow plot but save the data.
        print("  > Skipping imshow plot for 68x68 source matrix (data is saved).")
        
        # --- Save Individual Global Metrics (Source) ---
        if all_subject_indices:
            global_metrics_df = pd.DataFrame({
                'subject_index': all_subject_indices,
                'Beta_PLV_Global_Efficiency': all_global_efficiencies
            })
            gmetrics_path = os.path.join(DERIVATIVES_ROOT, 'individual_global_metrics_SOURCE.csv')
            global_metrics_df.to_csv(gmetrics_path, index=False)
            print(f"  > Saved individual SOURCE global metrics to: {gmetrics_path}")

    except Exception as e:
        print(f"! CRITICAL ERROR in run_source_connectivity: {e}")
        import traceback
        traceback.print_exc()

    return


def run_source_correlation():
    """
    [NEW ANALYSIS - CORRECTED] This is a copy of 'run_correlation_analysis'
    It correlates BCI Accuracy (from LDA) with the new SOURCE-SPACE
    network metrics.
    
    (Version 2: Fixes plotting variable bug)
    """
    print("\n" + "="*50)
    print("[Analysis 3.3] Running SOURCE-SPACE Correlation Analysis")
    print("="*50)

    # 1. Load the individual decoding scores (from the LDA classifier)
    decoding_scores_file = os.path.join(DERIVATIVES_ROOT, 'individual_decoding_scores_LDA.csv')
    if not os.path.exists(decoding_scores_file):
        print("! ERROR: 'individual_decoding_scores_LDA.csv' not found.")
        print("! Please run Decoding (option 7) first.")
        return
    decoding_df = pd.read_csv(decoding_scores_file)

    # 2. Load the individual SOURCE global metrics
    gmetrics_file = os.path.join(DERIVATIVES_ROOT, 'individual_global_metrics_SOURCE.csv')
    if not os.path.exists(gmetrics_file):
        print("! ERROR: 'individual_global_metrics_SOURCE.csv' not found.")
        print("! Please run Source Connectivity (option 15) first.")
        return
    global_metrics_df = pd.read_csv(gmetrics_file)

    # 3. Load and aggregate all individual SOURCE NODAL metrics
    all_roi_strengths = []
    subjects_in_order = decoding_df['subject_index'].astype(str).str.zfill(3)
    
    for subject_id in subjects_in_order:
        subject_dir = os.path.join(DERIVATIVES_ROOT, f'sub-{subject_id}')
        nodal_file = os.path.join(
            subject_dir, 
            f'sub-{subject_id}_desc-Beta_PLV_SOURCE_nodal_metrics.csv' # <-- NEW FILE
        )
        
        if not os.path.exists(nodal_file):
            print(f"  > Warning: Missing SOURCE nodal file for sub-{subject_id}.")
            all_roi_strengths.append(np.nan)
            continue
            
        nodal_df = pd.read_csv(nodal_file)
        
        try:
            # Example: Find the strength of the Left Postcentral Gyrus
            roi_s = nodal_df.loc[nodal_df['Channel'] == 'postcentral-lh', 'Nodal_Strength'].values
            all_roi_strengths.append(roi_s[0] if roi_s.size > 0 else np.nan)
        except Exception as e:
            print(f"  > Error processing SOURCE nodal file for {subject_id}: {e}")
            all_roi_strengths.append(np.nan)

    # 4. Combine all data into one DataFrame
    final_df = pd.merge(decoding_df, global_metrics_df, on='subject_index', how='left')
    final_df['postcentral-lh_Strength'] = all_roi_strengths
    
    final_df_clean = final_df.dropna()
    print(f"  > Correlating SOURCE metrics for {len(final_df_clean)} subjects.")

    if len(final_df_clean) < 2:
        print("! ERROR: Not enough data to run source correlation.")
        return

    print("\n--- SOURCE Correlation Analysis Results (Pearson's r) ---")
    
    # 5. Run and print the statistical correlations

    # --- START OF CORRECTION ---
    
    # Correlation: Accuracy vs. SOURCE Global Efficiency
    # **** SAVE THESE IN UNIQUE VARIABLES ****
    corr_glob_eff, p_val_glob_eff = pearsonr(final_df_clean['accuracy'], final_df_clean['Beta_PLV_Global_Efficiency'])
    print(f"Accuracy vs. SOURCE Global Efficiency (Beta PLV): r = {corr_glob_eff:.3f}, p = {p_val_glob_eff:.4f}")

    # Correlation: Accuracy vs. SOURCE ROI (Example)
    corr_roi, p_val_roi = pearsonr(final_df_clean['accuracy'], final_df_clean['postcentral-lh_Strength'])
    print(f"Accuracy vs. 'postcentral-lh' Nodal Strength (Beta PLV): r = {corr_roi:.3f}, p = {p_val_roi:.4f}")
    
    print("--------------------------------------------------\n")
    
    final_stats_file = os.path.join(DERIVATIVES_ROOT, 'individual_all_metrics_and_scores_SOURCE.csv')
    final_df_clean.to_csv(final_stats_file, index=False)
    print(f"  > Final SOURCE correlation data saved to {final_stats_file}")

    # --- PLOT: SOURCE-SPACE CORRELATION ---
    try:
        fig_corr, ax_corr = plt.subplots(figsize=(7, 5))
        ax_corr.scatter(final_df_clean['Beta_PLV_Global_Efficiency'], final_df_clean['accuracy'], 
                        alpha=0.7, edgecolors='k')
        
        # Add trend line
        z = np.polyfit(final_df_clean['Beta_PLV_Global_Efficiency'], final_df_clean['accuracy'], 1)
        p = np.poly1d(z)
        ax_corr.plot(final_df_clean['Beta_PLV_Global_Efficiency'], p(final_df_clean['Beta_PLV_Global_Efficiency']), "r--")
        
        # **** USE THE CORRECT VARIABLES IN THE TITLE ****
        ax_corr.set_title(f"Source-Space Correlation (r={corr_glob_eff:.3f}, p={p_val_glob_eff:.4f})")
        
        ax_corr.set_xlabel("Beta-PLV Global Efficiency (Source Space)")
        ax_corr.set_ylabel("BCI Decoding Accuracy (LDA)")
        ax_corr.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        plt.tight_layout()
        fig_corr.savefig(os.path.join(FIGURES_ROOT, 'report_correlation_accuracy_vs_source_glob_eff.png'), dpi=300)
        plt.show(block=False)
    except Exception as e:
        print(f"! FAILED to plot source correlation: {e}")

    # --- END OF CORRECTION ---
    
    return    
    
# ====================================================================
# --- III. MAIN EXECUTION AND MENU ---
# ====================================================================

# --- REPLACE your old main_menu function with this ---

def main_menu():
    """Provides a menu to select which part of the pipeline to run."""
    
    if not SUBJECTS_LIST:
        print("Cannot run pipeline. Check BIDS_ROOT path and BIDS structure.")
        return

    # Initialize summary results
    summary_results = {}
    
    while True:
        print("\n" + "="*50)
        print("       EEG Postural State Analysis Menu")
        print("="*50)
        print("1: **RUN FULL PIPELINE** (Preprocessing + All Analyses)")
        print("-" * 50)
        print("--- Primary Sensor-Space Pipeline ---")
        print("2: Preprocessing (Saves epochs, required for all others)")
        print("3: Connectivity (Requires 2, computes PLV/Coherence & Indiv. Metrics)")
        print("4: Graph Theory (Beta PLV) (Requires 3)")
        print("5: Granger Causality (Requires 2, computes directed flow)")
        print("6: NBS Analysis (Beta PLV) (Requires 2)")
        print("7: Decoding (Requires 2, CSP-LDA & Indiv. Scores)")
        print("-" * 50)
        print("--- Secondary Sensor-Space Analyses ---")
        print("8: Graph Theory (Beta Coherence) (Requires 3)")
        print("9: NBS Analysis (Beta Coherence) (Requires 2)")
        print("10: Graph Theory (Mu PLV) (Requires 3)")
        print("11: NBS Analysis (Mu PLV) (Requires 2)")
        print("12: RUN SENSOR CORRELATION (Requires 3 and 7)")
        print("13: STATISTICAL NETWORK TEST (MI vs. Idle) (Requires 2)")
        print("-" * 50)
        print("--- Source-Space Validation Pipeline ---")
        print("14: Run Source Modeling (Requires 2. Time-consuming)")
        print("15: Run Source Connectivity (Requires 14)")
        print("16: RUN SOURCE CORRELATION (Requires 7 and 15)")
        print("-" * 50)
        print("0: Exit")
        
        choice = input("Enter your choice (0-16): ").strip()

        # --- NEW: Status Tracker for Full Pipeline Run ---
        run_summary = [] # This list will store the status of each step
        
        def run_step(step_name, function_to_call, *args):
            """Helper function to run a step and log its status."""
            print(f"\n--- [RUNNING]: {step_name} ---")
            try:
                # Call the function (e.g., run_preprocessing)
                result = function_to_call(*args)
                
                # Check for explicit failure (e.g., run_preprocessing returns False)
                if result is False: 
                    raise Exception("Function returned False.")
                
                run_summary.append(f"✅ {step_name}: SUCCESS")
                return result # Return the result (e.g., summary_results)
            
            except Exception as e:
                print(f"\n! --- FAILED: {step_name} --- !")
                print(f"! ERROR: {e}")
                import traceback
                traceback.print_exc() # More details
                run_summary.append(f"❌ {step_name}: FAILED (Error: {e})")
                return None # Signal failure
        # --- END OF NEW BLOCK ---

        
        if choice == '0':
            print("Exiting pipeline. Goodbye!")
            break
        
        # --- Dependency Check and Execution ---
        
        # 2: Preprocessing
        if choice == '1' or choice == '2':
            if choice == '1':
                # Use the tracker for full run
                if run_step("Preprocessing", run_preprocessing, SUBJECTS_LIST) is None:
                    break # Stop the full run on critical failure
            else:
                # Run normally
                if not run_preprocessing(SUBJECTS_LIST):
                    print("Preprocessing failed. Cannot proceed with analysis.")
                    continue
            print("Preprocessing finished successfully.")


        # --- Load Data for Analysis Steps ---
        # (Modified to include new choices)
        if choice in ['1', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']:
            
            # If a later step is chosen, try to load pre-processed data
            if choice not in ['1', '2']:
                if not load_epochs(SUBJECTS_LIST):
                    print("ERROR: Cannot find pre-processed data (step 2) to continue.")
                    continue
            
            # If '12' or '16' is chosen, we don't need to re-load epochs_data
            if choice not in ['12', '16']:
                epochs_data = RESULTS_CACHE['all_epochs_data']
                ch_names = RESULTS_CACHE['ch_names']
                sfreq = RESULTS_CACHE['sfreq']
            
            # --- Check for Connectivity results (Steps 3, 4, 8, 10) ---
            if choice in ['1', '3', '4', '8', '10']:
                
                run_conn = False
                if choice in ['4', '8', '10'] and (
                    RESULTS_CACHE.get('avg_beta_plv') is None or 
                    RESULTS_CACHE.get('avg_beta_coh') is None or 
                    RESULTS_CACHE.get('avg_mu_plv') is None
                ):
                    print(f"Step {choice} requires Connectivity results. Running Connectivity (3) first...")
                    run_conn = True
                elif choice == '3':
                    run_conn = True
                elif choice == '1': # Part of full pipeline
                    run_conn = True

                if run_conn:
                    if choice == '1':
                        # Use tracker for full run
                        result = run_step("Connectivity", run_connectivity, epochs_data, ch_names, sfreq)
                        if result is None: break # Stop on critical failure
                        RESULTS_CACHE.update(result)
                    else:
                        # Run normally
                        connectivity_results = run_connectivity(epochs_data, ch_names, sfreq)
                        if connectivity_results:
                            RESULTS_CACHE.update(connectivity_results)
                        else:
                            print("! Connectivity failed to return results.")

            # --- Run Selected Analysis ---
            
            # 4: Graph Theory (Beta PLV)
            if choice == '1' or choice == '4':
                if RESULTS_CACHE.get('avg_beta_plv') is not None:
                    if choice == '1':
                        result = run_step("Graph Theory (Beta PLV)", run_graph_theory, 
                                          RESULTS_CACHE['avg_beta_plv'], "Beta_PLV", ch_names, summary_results)
                        if result: summary_results = result
                    else:
                        summary_results = run_graph_theory(
                            RESULTS_CACHE['avg_beta_plv'], "Beta_PLV", ch_names, summary_results)
                elif choice == '4':
                       print("Skipping Graph Theory: Run Connectivity (3) first.")

            # 5: Granger Causality
            if choice == '1' or choice == '5':
                if choice == '1':
                    run_step("Granger Causality", run_granger_causality, epochs_data, ch_names, sfreq)
                else:
                    run_granger_causality(epochs_data, ch_names, sfreq)

            # 6: NBS Analysis (Beta PLV)
            if choice == '1' or choice == '6':
                if choice == '1':
                    result = run_step("NBS (Beta PLV)", run_nbs_analysis, 
                                      epochs_data, sfreq, summary_results, 'plv', 
                                      FREQUENCY_BANDS['beta'][0], FREQUENCY_BANDS['beta'][1], "Beta_PLV")
                    if result: summary_results = result
                else:
                    summary_results = run_nbs_analysis(
                        epochs_data, sfreq, summary_results, 'plv', 
                        FREQUENCY_BANDS['beta'][0], FREQUENCY_BANDS['beta'][1], "Beta_PLV")

            # 7: Decoding (NEW - BENCHMARKING)
            if choice == '1' or choice == '7':
                if choice == '1':
                    result = run_step("Decoding", run_decoding, epochs_data, summary_results, ICA_RANDOM_STATE, DERIVATIVES_ROOT, FIGURES_ROOT, RESULTS_CACHE)
                    if result is None: break # Stop on critical failure
                    summary_results = result
                else:
                    # Note: run_decoding now runs all classifiers
                    summary_results = run_decoding(epochs_data, summary_results, ICA_RANDOM_STATE, DERIVATIVES_ROOT, FIGURES_ROOT, RESULTS_CACHE)

            # 8: Graph Theory (Beta Coherence)
            if choice == '1' or choice == '8':
                if RESULTS_CACHE.get('avg_beta_coh') is not None:
                    if choice == '1':
                        result = run_step("Graph Theory (Beta Coherence)", run_graph_theory, 
                                          RESULTS_CACHE['avg_beta_coh'], "Beta_Coherence", ch_names, summary_results)
                        if result: summary_results = result
                    else:
                        summary_results = run_graph_theory(
                            RESULTS_CACHE['avg_beta_coh'], "Beta_Coherence", ch_names, summary_results)
                elif choice == '8':
                       print("Skipping Graph Theory (Beta Coherence): Run Connectivity (3) first.")

            # 9: NBS Analysis (Beta Coherence)
            if choice == '1' or choice == '9':
                if choice == '1':
                    result = run_step("NBS (Beta Coherence)", run_nbs_analysis, 
                                      epochs_data, sfreq, summary_results, 'coh', 
                                      FREQUENCY_BANDS['beta'][0], FREQUENCY_BANDS['beta'][1], "Beta_Coherence")
                    if result: summary_results = result
                else:
                    summary_results = run_nbs_analysis(
                        epochs_data, sfreq, summary_results, 'coh', 
                        FREQUENCY_BANDS['beta'][0], FREQUENCY_BANDS['beta'][1], "Beta_Coherence")

            # 10: Graph Theory (Mu PLV)
            if choice == '1' or choice == '10':
                if RESULTS_CACHE.get('avg_mu_plv') is not None:
                    if choice == '1':
                        result = run_step("Graph Theory (Mu PLV)", run_graph_theory, 
                                          RESULTS_CACHE['avg_mu_plv'], "Mu_PLV", ch_names, summary_results)
                        if result: summary_results = result
                    else:
                        summary_results = run_graph_theory(
                            RESULTS_CACHE['avg_mu_plv'], "Mu_PLV", ch_names, summary_results)
                elif choice == '10':
                       print("Skipping Graph Theory (Mu PLV): Run Connectivity (3) first.")
            
            # 11: NBS Analysis (Mu PLV)
            if choice == '1' or choice == '11':
                if choice == '1':
                    result = run_step("NBS (Mu PLV)", run_nbs_analysis, 
                                      epochs_data, sfreq, summary_results, 'plv', 
                                      FREQUENCY_BANDS['mu'][0], FREQUENCY_BANDS['mu'][1], "Mu_PLV")
                    if result: summary_results = result
                else:
                    summary_results = run_nbs_analysis(
                        epochs_data, sfreq, summary_results, 'plv', 
                        FREQUENCY_BANDS['mu'][0], FREQUENCY_BANDS['mu'][1], "Mu_PLV")

            # 12: Correlation Analysis (Sensor)
            if choice == '1' or choice == '12':
                if choice == '1':
                    run_step("Sensor Correlation", run_correlation_analysis)
                else:
                    run_correlation_analysis()
                
            # 13: Statistical Network Comparison
            if choice == '1' or choice == '13':
                if choice == '1':
                    run_step("Statistical Network Test", run_MI_vs_Idle_network_comparison)
                else:
                    run_MI_vs_Idle_network_comparison()

            # --- NEW MENU OPTIONS ---
            
            # 14: Source Modeling
            if choice == '14':
                # This is a standalone step, not part of the full pipeline (1)
                run_source_modeling()
                
            # 15: Source Connectivity
            if choice == '15':
                if 'all_label_epochs' not in RESULTS_CACHE:
                    print("! Running Source Modeling (14) first...")
                    if not run_source_modeling():
                        print("! Source Modeling failed. Cannot run Source Connectivity.")
                        continue # Stop this choice
                
                run_source_connectivity()

            # 16: Source Correlation
            if choice == '16':
                run_source_correlation()
                
            # --- END NEW OPTIONS ---

            # --- Save Final Summary if any analysis was performed ---
            if summary_results and choice!= '0':
                try:
                    summary_df = pd.DataFrame.from_dict(summary_results, orient='index').transpose()
                    csv_path_summary = os.path.join(DERIVATIVES_ROOT, 'group_summary_results.csv')
                    summary_df.to_csv(csv_path_summary, index=False)
                    print(f"\n--- Saved updated summary results to: {csv_path_summary} ---")
                except Exception as e:
                    print(f"\n! FAILED to save summary CSV: {e}")

            # --- NEW: Print Full Run Summary ---
            if choice == '1' and run_summary:
                print("\n" + "="*50)
                print("         FULL PIPELINE RUN SUMMARY")
                print("="*50)
                for line in run_summary:
                    print(f"  {line}")
                print("="*50)
            # --- END OF NEW BLOCK ---

            if choice!= '0':
                print("\n--- Finished selected task(s). ---")
                plt.show() # Block until figures are closed
        
        elif choice == '0':
            pass # Already handled
        
        else:
            print(f"Invalid choice: {choice}. Please enter a number between 0 and 16.")
            
if __name__ == "__main__":
    main_menu()